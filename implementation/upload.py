
from pathlib import Path
import shutil
import uuid

# Project's root directory
BASE_DIR = Path(__file__).parent.parent

# Directory to store uploaded files
UPLOAD_DIR = BASE_DIR / "uploaded_documents"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(file_paths):

    saved_files = []

    for file_path in file_paths:

        # Original uploaded file
        source = Path(file_path)

        # Generate unique filename
        unique_name = f"{uuid.uuid4().hex}_{source.name}"

        # Permanent destination
        destination = UPLOAD_DIR / unique_name

        # Copy the file
        shutil.copy2(source, destination)

        saved_files.append(str(destination.resolve()))

    return saved_files
