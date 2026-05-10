from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download


REPO_ID = "SilpaCS/kneeosteoarthritis"
FILENAME = "data.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EXTRACT_DIR = DATA_DIR / "kneeosteoarthritis"


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )

    return Path(downloaded_path)


def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Extraction skipped!!! Directory already exists: {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Extracted: {extract_dir}")


def main() -> None:
    print(f"Downloading : {REPO_ID}")
    downloaded_path = download_dataset()
    print(f"Downloaded : {downloaded_path}")

    extract_dataset(downloaded_path, EXTRACT_DIR)


if __name__ == "__main__":
    main()