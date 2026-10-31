
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

UPLOADED_DIR = BASE_DIR / "uploaded_documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"


def clear_all():

    # 1. Delete all uploaded documents
    if UPLOADED_DIR.exists():
        shutil.rmtree(UPLOADED_DIR)

    UPLOADED_DIR.mkdir(parents=True, exist_ok=True)

    # # 2. Delete the complete ChromaDB directory
    # if VECTOR_DB_DIR.exists():
    #     shutil.rmtree(VECTOR_DB_DIR)

    # # 3. Recreate an empty directory
    # VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    return ("- Uploaded documents deleted.\n")


if __name__ == "__main__":
    print(clear_all())
