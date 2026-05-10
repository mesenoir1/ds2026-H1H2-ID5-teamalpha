import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "kneeosteoarthritis"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def write_split_csv(csv_path: Path, paths: list[str], labels: list[int]) -> None:
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])

        for path, label in zip(paths, labels):
            writer.writerow([path, label])


def find_images(data_dir: Path) -> list[Path]:
    return [
        path for path in data_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def infer_label_from_path(path: Path) -> int:
    label = path.parent.name

    if label in {"0", "1", "2", "3", "4"}:
        return int(label)

    raise ValueError(f"Could not infer KL label from parent directory: {path}")

def main() -> None:
    print(f"Lade Bildpfade aus {DATA_DIR}...")
    image_paths = find_images(DATA_DIR)
    
    if not image_paths:
        print("No images found.")
        return

    # relative paths
    paths_str = [str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in image_paths]
    labels = [infer_label_from_path(p) for p in image_paths]

    # Split - 70% Training / 30% Rest
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths_str, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    # Split Rest - 15% Validation / 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    write_split_csv(SPLIT_DIR / "train.csv", X_train, y_train)
    write_split_csv(SPLIT_DIR / "val.csv", X_val, y_val)
    write_split_csv(SPLIT_DIR / "test.csv", X_test, y_test)
    
    print(f"Trainingdata:  {len(X_train)} images")
    print(f"Validationdata: {len(X_val)} images")
    print(f"Testdata:       {len(X_test)} images")

    # # report CSV Data
    # csv_path = FIGURE_DIR / "dataset_split.csv"
    
    # with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["filepath", "label", "split"]) # Header
        
    #     for path, label in zip(X_train, y_train):
    #         writer.writerow([path, label, "train"])
    #     for path, label in zip(X_val, y_val):
    #         writer.writerow([path, label, "val"])
    #     for path, label in zip(X_test, y_test):
    #         writer.writerow([path, label, "test"])
            
    # print(f"\nStored Split at: {csv_path}")

if __name__ == "__main__":
    main()