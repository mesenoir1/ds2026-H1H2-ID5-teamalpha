from pathlib import Path
from collections import Counter, defaultdict
import random

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "kneeosteoarthritis"
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


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


def plot_class_distribution(counts: Counter) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    labels = sorted(counts)

    plt.figure()
    plt.bar([str(label) for label in labels], [counts[label] for label in labels])
    plt.title("Class Distribution")
    plt.xlabel("KL Grade")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=200)
    plt.close()


def plot_sample_images(image_paths: list[Path], seed: int = 42) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    paths_by_label = defaultdict(list)

    for path in image_paths:
        label = infer_label_from_path(path)
        paths_by_label[label].append(path)

    plt.figure(figsize=(12, 4))

    for idx, label in enumerate(sorted(paths_by_label)):
        image_path = random.choice(paths_by_label[label])
        image = Image.open(image_path).convert("L")

        plt.subplot(1, 5, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"KL {label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sample_images_by_class.png", dpi=200)
    plt.close()


def audit_image_sizes(image_paths: list[Path], sample_size: int = 100, seed: int = 42) -> Counter:
    random.seed(seed)
    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))

    sizes = Counter()

    for path in sample_paths:
        with Image.open(path) as img:
            sizes[img.size] += 1

    return sizes


def main() -> None:
    print(f"Reading from: {DATA_DIR}")

    image_paths = find_images(DATA_DIR)

    if not image_paths:
        raise RuntimeError(
            f"No images found in {DATA_DIR}. "
            "Check dataset_download"
        )

    print(f"Total images found: {len(image_paths)}")

    labels = [infer_label_from_path(path) for path in image_paths]
    counts = Counter(labels)

    print("\nClass distribution:")
    for label in sorted(counts):
        print(f"KL grade {label}: {counts[label]}")

    plot_class_distribution(counts)
    plot_sample_images(image_paths)

    sizes = audit_image_sizes(image_paths)

    print("\nImage size sample:")
    for size, count in sizes.items():
        print(f"{size}: {count}")

    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()