import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from densenet_dataset import KneeOADataset # private import 


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVAL_DIR = PROJECT_ROOT / "data" / "eval_elina"
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"

MODELS_TO_EVALUATE = ["densenet_ce", "densenet_weighted_ce"]
CLASS_LABELS = [0, 1, 2, 3, 4]


def build_densenet201(num_classes: int = 5) -> nn.Module:
    """Baut exakt dieselbe Architektur auf wie im Training."""
    model = models.densenet201(weights=None) # Keine Weights nötig, da wir unsere eigenen laden
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path, device):
    """Lädt das Modell und die trainierten Gewichte."""
    model = build_densenet201(num_classes=5)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def evaluate(model_name, device, dataloader):
    print(f"\nStarte Evaluierung für: {model_name}")
    model_path = OUTPUT_DIR / model_name / "best_model.pt"

    if not model_path.exists():
        print(f"WARNUNG: Modell nicht gefunden unter {model_path}. Überspringe...")
        return None

    model = load_model(model_path, device)

    y_true = []
    y_pred = []
    prediction_rows = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)

            batch_paths = dataloader.dataset.data.iloc[
                sample_index: sample_index + batch_size
            ]["image_path"].tolist()

            for image_path, true_label, pred_label, probs in zip(
                batch_paths,
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                probabilities.cpu().numpy(),
            ):
                row = {
                    "image_path": image_path,
                    "true_label": int(true_label),
                    "pred_label": int(pred_label),
                    "correct": int(true_label == pred_label),
                }

                for class_idx, prob in enumerate(probs):
                    row[f"prob_{class_idx}"] = float(prob)

                prediction_rows.append(row)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            sample_index += batch_size

    # 1. Metriken berechnen & speichern
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    eval_df = pd.DataFrame(report).transpose()

    summary_row = {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(y_true),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    csv_out_path = EVAL_DIR / f"eval_{model_name}.csv"
    eval_df.to_csv(csv_out_path)
    print(f"Evaluierung gespeichert: {csv_out_path}")

    summary_out_path = EVAL_DIR / f"summary_{model_name}.csv"
    pd.DataFrame([summary_row]).to_csv(summary_out_path, index=False)
    print(f"Summary gespeichert: {summary_out_path}")

    predictions_out_path = EVAL_DIR / f"predictions_{model_name}.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_out_path, index=False)
    print(f"Predictions gespeichert: {predictions_out_path}")

    # 2. Confusion Matrix erstellen & speichern
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True KL Grade")
    plt.xlabel("Predicted KL Grade")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_out_path = FIGURE_DIR / f"cm_elina_{model_name}.png"
    plt.savefig(fig_out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix gespeichert: {fig_out_path}")

    return summary_row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Device: {device}")

    test_dataset = KneeOADataset(TEST_CSV, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    summaries = []

    for model_name in MODELS_TO_EVALUATE:
        summary = evaluate(model_name, device, test_loader)

        if summary is not None:
            summaries.append(summary)

    if summaries:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        comparison_out_path = EVAL_DIR / "model_comparison.csv"
        pd.DataFrame(summaries).to_csv(comparison_out_path, index=False)
        print(f"\nModel comparison gespeichert: {comparison_out_path}")


if __name__ == "__main__":
    main()