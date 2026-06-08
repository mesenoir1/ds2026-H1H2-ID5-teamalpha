import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "code"))

try:
    from roi_dynamic import get_roi_mask
except ImportError:
    print("FEHLER: roi_dynamic.py konnte nicht importiert werden. Stelle sicher, dass sie im code/ Ordner liegt.")
    sys.exit(1)

# MODEL ARCHITECTURE (Compability) ===
def build_densenet201(num_classes: int = 5) -> nn.Module:
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

class DenseNetWithFeatures(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base_model = models.densenet201(weights=None)
        self.features = base_model.features
        self.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits, features

# === GRAD-CAM ===
class SimpleGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Target Layer last Feature-Block in DenseNet
        target_layer = self.model.features
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        if isinstance(logits, tuple): 
            logits = logits[0]
            
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

# === ELPER FUNCTIONS ===
def create_border_mask(h, w, border_frac):
    mask = np.zeros((h, w), dtype=np.float32)
    by = max(1, int(h * border_frac))
    bx = max(1, int(w * border_frac))
    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1
    return mask

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    arch = config.get("architecture", "")

    if arch == "DenseNetWithFeatures":
        model = DenseNetWithFeatures(num_classes=5)
    else:
        model = build_densenet201(num_classes=5)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# === MAIN SCRIPT ===
def evaluate_model_faithfulness(model_name, args, device):
    print(f"\nStarte Faithfulness Evaluierung für Modell: {model_name}")
    model_path = PROJECT_ROOT / "outputs" / model_name / "best_model.pt"
    
    if not model_path.exists():
        print(f"WARNUNG: Checkpoint nicht gefunden unter {model_path}. Überspringe.")
        return None

    model = load_model(model_path, device)
    grad_cam = SimpleGradCAM(model)

    df = pd.read_csv(PROJECT_ROOT / args.test_csv)
    
    # Label CLMN FINDER
    label_col = "label" if "label" in df.columns else "true_label"
    if label_col not in df.columns and "KL_Grade" in df.columns:
        label_col = "KL_Grade"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    roi_scores = []
    border_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {model_name}"):
        img_path = PROJECT_ROOT / row["image_path"] if "image_path" in row else PROJECT_ROOT / row["file_name"]
        
        # load and prepare for roi mask
        img_pil = Image.open(img_path).convert("L").resize((224, 224), Image.BILINEAR)
        img_np = np.asarray(img_pil).astype(np.uint8)
        
        # prepare
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # prediction for gradcam
        with torch.no_grad():
            logits = model(input_tensor)
            if isinstance(logits, tuple): logits = logits[0]
            pred_class = torch.argmax(logits, dim=1).item()
            
        # generate gradcam
        cam = grad_cam.generate(input_tensor, pred_class)
        cam_resized = cv2.resize(cam, (224, 224))
        
        # generate mask
        roi_mask = get_roi_mask(img_np)
        roi_mask_float = (roi_mask > 0).astype(np.float32)
        border_mask_float = create_border_mask(224, 224, args.border_frac)
        
        # Faithfulness Scores 
        total_attention = np.sum(cam_resized) + 1e-8
        attention_in_roi = np.sum(cam_resized * roi_mask_float)
        attention_in_border = np.sum(cam_resized * border_mask_float)
        
        roi_scores.append(attention_in_roi / total_attention)
        border_scores.append(attention_in_border / total_attention)

    # Results
    mean_roi = np.mean(roi_scores)
    mean_border = np.mean(border_scores)
    
    print(f"Ergebnis {model_name}:")
    print(f" -> ROI Faithfulness: {mean_roi*100:.2f}% der Attention liegt in der ROI")
    print(f" -> Border Attention: {mean_border*100:.2f}% der Attention liegt im Randbereich")
    
    return {
        "model": model_name,
        "mean_roi_faithfulness": mean_roi,
        "mean_border_attention": mean_border,
        "border_fraction_used": args.border_frac,
        "num_samples": len(df)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="List of Models")
    parser.add_argument("--test-csv", default="data/splits/test.csv", help="Test CSV")
    parser.add_argument("--border-frac", type=float, default=0.08, help="border thickness")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Device: {device}")

    eval_dir = PROJECT_ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name in args.models:
        res = evaluate_model_faithfulness(model_name, args, device)
        if res:
            results.append(res)
   
            pd.DataFrame([res]).to_csv(eval_dir / f"faithfulness_{model_name}.csv", index=False)

    if len(results) > 1:
        # Eine aggregierte CSV speichern, falls mehrere Modelle verglichen werden
        pd.DataFrame(results).to_csv(eval_dir / "faithfulness_comparison.csv", index=False)
        print(f"\nSave summary under {eval_dir / 'faithfulness_comparison.csv'}")

if __name__ == "__main__":
    main()