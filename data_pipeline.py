"""
Data processing pipeline for extracting metadata and generating semantic embeddings.

This script processes a directory of files (PDFs, audio, images), extracts their
metadata, saves it to JSON sidecar files, generates embeddings from the
metadata, and caches those embeddings in a FAISS index for fast retrieval.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from mutagen import File as MutagenFile
from txtai.embeddings import Embeddings

# --- Configuration ---
# Set the source directory for files to process. Can be overridden by an environment variable.
SOURCE_DIRECTORY = Path(os.getenv("SOURCE_DIRECTORY", "source_data"))
# Path to the FAISS index file
INDEX_PATH = Path("embeddings.faiss")
# Log file name
LOG_FILE = "pipeline.log"
# Pre-trained model for generating embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Logger Setup ---
# Configure logging to output to both a file and the console.
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File handler for logging to a file
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler for logging to the console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def extract_pdf_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        A dictionary of metadata or None if an error occurs.
    """
    try:
        with fitz.open(path) as doc:
            metadata = doc.metadata or {}
            metadata["page_count"] = doc.page_count
            return {k: str(v) for k, v in metadata.items()}
    except Exception as e:
        logger.error("Failed to extract PDF metadata from %s: %s", path, e)
        return None


def extract_audio_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from an audio file.

    Args:
        path: Path to the audio file.

    Returns:
        A dictionary of metadata or None if an error occurs.
    """
    try:
        audio = MutagenFile(path, easy=True)
        if not audio:
            return None
        metadata = dict(audio)
        if hasattr(audio, 'info'):
            metadata["duration_seconds"] = getattr(audio.info, 'length', None)
            metadata["bitrate"] = getattr(audio.info, 'bitrate', None)
        return {k: str(v) for k, v in metadata.items()}
    except Exception as e:
        logger.error("Failed to extract audio metadata from %s: %s", path, e)
        return None


def extract_image_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from an image file using ExifTool."""
    try:
        result = subprocess.run(
            ["exiftool", "-j", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            logger.error("ExifTool failed for %s with error: %s", path, result.stderr.strip())
            return None
        return json.loads(result.stdout)[0]
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to extract image metadata from %s: %s", path, e)
        return None


def process_file(path: Path, embeddings: Embeddings):
    """Process a single file: extract metadata, save sidecar, cache embedding."""
    file_id = str(path.resolve())

    if embeddings.count() > 0 and embeddings.search(f"id:{file_id}", limit=1):
        logger.debug("Skipping already indexed file: %s", path.name)
        return

    logger.info("Processing new file: %s", path.name)
    metadata = None
    ext = path.suffix.lower()

    if ext == ".pdf":
        metadata = extract_pdf_metadata(path)
    elif ext in {".mp3", ".wav", ".flac", ".m4a"}:
        metadata = extract_audio_metadata(path)
    elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"}:
        metadata = extract_image_metadata(path)
    else:
        logger.debug("Skipping unsupported file type: %s", path.name)
        return

    if not metadata:
        logger.warning("No metadata could be extracted for %s.", path.name)
        return

    sidecar_path = path.with_suffix(path.suffix + ".json")
    try:
        serializable_metadata = {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, list, dict, bool, type(None)))
        }
        sidecar_path.write_text(json.dumps(serializable_metadata, indent=4))
        logger.info("Wrote metadata sidecar to %s", sidecar_path.name)
    except TypeError as e:
        logger.error("Failed to write non-serializable metadata for %s: %s", path.name, e)
        return
    except Exception as e:
        logger.error("Failed to write sidecar file for %s: %s", path.name, e)
        return

    metadata_text = json.dumps(serializable_metadata)
    try:
        embeddings.upsert([(file_id, metadata_text, None)])
        logger.info("Generated and cached embedding for %s", path.name)
    except Exception as e:
        logger.error("Failed to generate embedding for %s: %s", path.name, e)


# Optional: uncomment for ChromaDB caching alternative
# def get_chroma_client(db_path: str = "./chroma_db"):
#     import chromadb
#     """Return a ChromaDB client instance."""
#     return chromadb.PersistentClient(path=db_path)


def main():
    """Run the data processing pipeline."""
    logger.info("--- Starting Data Pipeline ---")

    if not SOURCE_DIRECTORY.exists():
        logger.info("Source directory '%s' not found. Creating it.", SOURCE_DIRECTORY)
        SOURCE_DIRECTORY.mkdir(parents=True)
        logger.info("Please add files to '%s' and run the script again.", SOURCE_DIRECTORY)
        return

    logger.info("Initializing embeddings model: %s", MODEL_NAME)
    try:
        embeddings = Embeddings({"path": MODEL_NAME, "content": True})
    except Exception as e:
        logger.critical("Failed to initialize embeddings model: %s", e)
        return

    if INDEX_PATH.exists():
        try:
            embeddings.load(str(INDEX_PATH))
            logger.info(
                "Successfully loaded existing FAISS index with %d entries.",
                embeddings.count(),
            )
        except Exception as e:
            logger.error(
                "Found an index file at '%s' but failed to load it: %s. A new one will be created.",
                INDEX_PATH,
                e,
            )
    else:
        logger.info("No existing FAISS index found at '%s'. A new one will be created.", INDEX_PATH)

    logger.info("Scanning directory: %s", SOURCE_DIRECTORY.resolve())
    initial_count = embeddings.count()

    files_to_process = [
        p for p in SOURCE_DIRECTORY.rglob("*") if p.is_file() and not p.name.endswith('.json')
    ]
    logger.info("Found %d total files to check.", len(files_to_process))

    for path in files_to_process:
        process_file(path, embeddings)

    final_count = embeddings.count()
    if final_count > initial_count:
        try:
            embeddings.save(str(INDEX_PATH))
            logger.info(
                "Saved FAISS index with %d total entries to %s.",
                final_count,
                INDEX_PATH,
            )
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)
    else:
        logger.info("No new files were added to the index. No save operation needed.")

    logger.info("--- Pipeline Complete ---")


if __name__ == "__main__":
    main()
