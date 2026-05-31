import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
import argparse

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def generate_classwise_gradcam_grid(model, target_layers, input_dir, output_dir, samples_per_class):
    """
    Berechnet die durchschnittliche Grad-CAM pro Klasse und eine globale Durchschnitts-CAM
    und plottet alles übersichtlich in einem Grid auf einem Bild.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

  
    class_folders = sorted([f for f in input_path.iterdir() if f.is_dir()])
    if not class_folders:
        print("No subfolders found.")
        return

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers)

    class_heatmaps = {}
    global_heatmap = np.zeros((224, 224), dtype=np.float32)
    global_count = 0

    print(f"Starte Verarbeitung: {samples_per_class} Samples per class...")

  
    for folder in class_folders:
        class_name = folder.name
        images_in_folder = list(folder.glob("*.png"))
        
        n_draw = min(samples_per_class, len(images_in_folder))
        if n_draw == 0:
            continue
            
        selected_images = random.sample(images_in_folder, n_draw)
        
        class_accumulator = np.zeros((224, 224), dtype=np.float32)
        valid_samples = 0

        try:
            class_idx = int(class_name)
            targets = [ClassifierOutputTarget(class_idx)]
        except ValueError:
            targets = None

        print(f"processed class '{class_name}' ({n_draw} img)...")
        
        for img_path in selected_images:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                class_accumulator += grayscale_cam
                valid_samples += 1
                
            except Exception as e:
                pass 
                
        if valid_samples > 0:
            avg_class_cam = class_accumulator / valid_samples
            class_heatmaps[class_name] = avg_class_cam
            
            # glob akkumul
            global_heatmap += class_accumulator
            global_count += valid_samples

    if global_count == 0:
        print("Error: no images processed.")
        return


    global_avg_cam = global_heatmap / global_count

    n_plots = len(class_heatmaps) + 1
    cols = 3
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    def plot_cam_on_ax(ax, cam_matrix, title):
        norm_cam = (cam_matrix - np.min(cam_matrix)) / (np.max(cam_matrix) - np.min(cam_matrix) + 1e-8)
        im = ax.imshow(norm_cam, cmap='jet')
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis('off')
        return im

    for idx, (c_name, c_matrix) in enumerate(sorted(class_heatmaps.items())):
        im = plot_cam_on_ax(axes[idx], c_matrix, f"Class {c_name} (avg)")

    plot_cam_on_ax(axes[len(class_heatmaps)], global_avg_cam, f"Combined (All Classes)")

    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    # Speichern
    save_path = output_path / f"gradcam_grid_samples_{samples_per_class}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Done and saved: {save_path}")



if __name__ == "__main__":
    import torchvision.models as models
    import torch
    
    parser = argparse.ArgumentParser(description="Classwise Grad-CAM Grid Generator")
    parser.add_argument("-i", "--input", required=True, help="")
    parser.add_argument("-o", "--output", required=True, help="")
    parser.add_argument("-n", "--num", type=int, default=100, help="")
    parser.add_argument("-m", "--model_weights", required=True, help="")
    
    args = parser.parse_args()
    

    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 5) 
    
    checkpoint = torch.load(args.model_weights, map_location=torch.device('cpu'))
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    target_layers = [model.features[-1]]

    generate_classwise_gradcam_grid(model, target_layers, args.input, args.output, args.num)