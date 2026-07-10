from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from densenet_dataset import get_densenet_transforms
    from evaluate_faithfulness import DenseNetWithFeatures, build_densenet201
    from generate_gradcam import GradCAM
    from roi_dynamic import get_roi_mask
    from suspicious_cases_dynamic import attention_inside_mask, border_mask, make_overlay, resize_mask
except Exception as exc:  # pragma: no cover - shown in the Streamlit UI
    raise ImportError(
        "Could not import the existing paper implementation from code/. "
        "Run the app from the project root and install requirements.txt. "
        f"Original error: {exc}"
    ) from exc


CLASS_LABELS = [0, 1, 2, 3, 4]
ROI_THRESHOLD = 0.40
BORDER_THRESHOLD = 0.20
BORDER_FRAC = 0.08


@dataclass(frozen=True)
class CheckpointInfo:
    seed: int
    path: Path
    folder: Path
    config: dict


@dataclass
class PredictionResult:
    probabilities: np.ndarray
    pred_class: int
    confidence: float
    seed_rows: list[dict]


@dataclass
class ExplanationResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    roi_inside: float
    border_attention: float
    suspicious: bool
    suspicion_reason: str


class LogitsOnlyModel(nn.Module):
    """Adapter for existing final models whose forward returns (logits, features)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_config(folder: Path) -> dict:
    config_path = folder / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_seed(folder: Path, config: dict) -> int | None:
    training = config.get("training", {})
    for value in (config.get("seed"), training.get("seed")):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    name = folder.name.lower()
    patterns = [
        r"seed[_-]?(\d+)",
        r"smooth[_-]?(\d+)",
        r"(^|[_-])(\d{2})([_-]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(match.lastindex or 1))
    return None


def discover_checkpoints(root: str | Path) -> dict[int, CheckpointInfo]:
    root_path = resolve_path(root)
    if not root_path.exists():
        return {}

    checkpoints: dict[int, CheckpointInfo] = {}
    for checkpoint_path in sorted(root_path.rglob("best_model.pt")):
        folder = checkpoint_path.parent
        config = read_config(folder)
        seed = infer_seed(folder, config)
        if seed is None:
            continue
        checkpoints[seed] = CheckpointInfo(
            seed=seed,
            path=checkpoint_path,
            folder=folder,
            config=config,
        )
    return dict(sorted(checkpoints.items()))


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "model"):
        if key not in checkpoint:
            continue
        value = checkpoint[key]
        if isinstance(value, nn.Module):
            return value.state_dict()
        return value

    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _checkpoint_config(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return {}


def build_model_for_checkpoint(checkpoint_config: dict, model_family: str) -> nn.Module:
    architecture = str(checkpoint_config.get("architecture", "")).lower()
    if model_family == "final" or architecture == "densenetwithfeatures":
        return DenseNetWithFeatures(num_classes=5)
    return build_densenet201(num_classes=5)


def load_checkpoint_model(checkpoint_path: str | Path, model_family: str, device: torch.device) -> nn.Module:
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as first_exc:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            raise RuntimeError(f"Could not read checkpoint {checkpoint_path}: {first_exc}") from first_exc
        except Exception as second_exc:
            raise RuntimeError(f"Could not read checkpoint {checkpoint_path}: {second_exc}") from second_exc

    model = build_model_for_checkpoint(_checkpoint_config(checkpoint), model_family=model_family)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint format was recognized, but weights did not load into the "
            f"{model.__class__.__name__} architecture: {checkpoint_path}. Error: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    image_gray = image.convert("L").resize((224, 224), Image.BILINEAR)
    image_np = np.asarray(image_gray).astype(np.uint8)
    transform = get_densenet_transforms("test")
    tensor = transform(image_gray).unsqueeze(0).to(device)
    return tensor, image_np


def logits_from_model(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    output = model(image_tensor)
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.no_grad()
def predict_probabilities(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    logits = logits_from_model(model, image_tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs.astype(np.float32)


def summarize_probabilities(probabilities: np.ndarray) -> tuple[int, float]:
    pred_class = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_class])
    return pred_class, confidence


def predict_single(model: nn.Module, image_tensor: torch.Tensor, seed: int) -> PredictionResult:
    probs = predict_probabilities(model, image_tensor)
    pred_class, confidence = summarize_probabilities(probs)
    return PredictionResult(
        probabilities=probs,
        pred_class=pred_class,
        confidence=confidence,
        seed_rows=[seed_prediction_row(seed, probs)],
    )


def predict_ensemble(
    checkpoints: dict[int, CheckpointInfo],
    model_family: str,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> PredictionResult:
    rows = []
    probabilities = []
    for seed, checkpoint in sorted(checkpoints.items()):
        model = load_checkpoint_model(checkpoint.path, model_family=model_family, device=device)
        probs = predict_probabilities(model, image_tensor)
        rows.append(seed_prediction_row(seed, probs))
        probabilities.append(probs)

    if not probabilities:
        raise ValueError(f"No checkpoints available for {model_family} ensemble prediction.")

    mean_probs = np.mean(np.stack(probabilities, axis=0), axis=0)
    pred_class, confidence = summarize_probabilities(mean_probs)
    return PredictionResult(
        probabilities=mean_probs,
        pred_class=pred_class,
        confidence=confidence,
        seed_rows=rows,
    )


def seed_prediction_row(seed: int, probabilities: np.ndarray) -> dict:
    pred_class, confidence = summarize_probabilities(probabilities)
    row = {
        "seed": seed,
        "predicted_KL": pred_class,
        "confidence": confidence,
    }
    for class_idx in CLASS_LABELS:
        row[f"prob_KL_{class_idx}"] = float(probabilities[class_idx])
    return row


def generate_predicted_class_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    logits_model = LogitsOnlyModel(model)
    target_layer = model.features.denseblock4 if hasattr(model.features, "denseblock4") else model.features
    gradcam = GradCAM(model=logits_model, target_layer=target_layer)
    try:
        heatmaps, logits, pred_classes = gradcam(image_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    pred_class = int(pred_classes[0].detach().cpu().item())
    heatmap = heatmaps[0].detach().cpu().numpy().astype(np.float32)
    return heatmap, torch.softmax(logits, dim=1)[0].detach().cpu().numpy(), pred_class


def make_masks(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi = get_roi_mask(image_np)
    roi = resize_mask(roi, image_np.shape)
    border = border_mask(image_np.shape[0], image_np.shape[1], border_frac=BORDER_FRAC)
    return roi.astype(np.uint8), border.astype(np.uint8)


def compute_alignment_metrics(heatmap: np.ndarray, roi: np.ndarray, border: np.ndarray) -> tuple[float, float]:
    roi_inside = float(attention_inside_mask(heatmap, roi))
    border_attention = float(attention_inside_mask(heatmap, border))
    return roi_inside, border_attention


def suspicious_reason(roi_inside: float, border_attention: float) -> tuple[bool, str]:
    reasons = []
    if roi_inside < ROI_THRESHOLD:
        reasons.append("ROIInside < 0.40")
    if border_attention > BORDER_THRESHOLD:
        reasons.append("BorderAttention > 0.20")
    return bool(reasons), "; ".join(reasons) if reasons else "not suspicious"


def explain_model(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image_np: np.ndarray,
    roi: np.ndarray,
    border: np.ndarray,
    target_class: int | None = None,
) -> ExplanationResult:
    heatmap, _, _ = generate_predicted_class_gradcam(model, image_tensor, target_class=target_class)
    return explain_heatmap(heatmap, image_np, roi, border)


def explain_heatmap(
    heatmap: np.ndarray,
    image_np: np.ndarray,
    roi: np.ndarray,
    border: np.ndarray,
) -> ExplanationResult:
    if heatmap.shape != image_np.shape:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    heatmap = np.maximum(heatmap.astype(np.float32), 0)
    heatmap_min = float(heatmap.min())
    heatmap_max = float(heatmap.max())
    if heatmap_max - heatmap_min > 1e-8:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    roi_inside, border_attention = compute_alignment_metrics(heatmap, roi, border)
    suspicious, reason = suspicious_reason(roi_inside, border_attention)
    overlay = make_overlay(image_np, heatmap, alpha=0.45)
    return ExplanationResult(
        heatmap=heatmap,
        overlay=overlay,
        roi_inside=roi_inside,
        border_attention=border_attention,
        suspicious=suspicious,
        suspicion_reason=reason,
    )


def mask_overlay(image_np: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    base = image_np.astype(np.float32)
    if base.max() > 1:
        base = base / 255.0
    rgb = np.stack([base, base, base], axis=-1)
    colored = np.zeros_like(rgb)
    colored[..., 0] = color[0]
    colored[..., 1] = color[1]
    colored[..., 2] = color[2]
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.40
    return np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)


def prediction_table(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for col in df.columns:
        if col.startswith("prob_") or col == "confidence":
            df[col] = df[col].map(lambda value: f"{value:.3f}")
    return df
