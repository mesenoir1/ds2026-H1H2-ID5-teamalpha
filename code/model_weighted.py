import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha_resNet/data/splits"
# Better portable version:
# DATA_DIR = os.path.join(BASE_DIR, "data", "splits")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
 
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# changed for weighted model output folder
# OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "resnet101_ce")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "resnet101_weighted_ce")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# changed
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor()
# ])
# Needed normalization for resnet model:

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class KneeXrayCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        required_columns = {"image_path", "label"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"CSV must contain columns: {required_columns}. "
                f"Found: {set(self.data.columns)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        label = int(self.data.iloc[idx]["label"])

        if not os.path.isabs(image_path):
            image_path = os.path.join(BASE_DIR,image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = KneeXrayCSVDataset(
    csv_file=TRAIN_CSV,
    transform=transform
)

val_dataset = KneeXrayCSVDataset(
    csv_file=VAL_CSV,
    transform=transform
)

test_dataset = KneeXrayCSVDataset(
    csv_file=TEST_CSV,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory = torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory = torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory = torch.cuda.is_available()
)


model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

# Replace final classification layer for 5 K-L grades
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

#shd chamge optimizer and scheduler?

# changed for weighted model:
# criterion = nn.CrossEntropyLoss()

# with weighted loss
class_counts = train_dataset.data["label"].value_counts().sort_index()

# Make sure every class 0-4 exists in the counts
class_counts = class_counts.reindex(range(NUM_CLASSES), fill_value=0)

if (class_counts == 0).any():
    raise ValueError(
        f"At least one class has zero samples in the training set: {class_counts.to_dict()}"
    )

class_weights = len(train_dataset) / (NUM_CLASSES * class_counts)

class_weights = torch.tensor(
    class_weights.values,
    dtype=torch.float32,
    device=DEVICE
)

print("Class counts:")
print(class_counts)

print("Class weights:")
print(class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels,all_preds,average="macro",zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1


def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels,all_preds,average="macro",zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1

# best model should be chosen by macro-f1 (same as densenet so for consistency)
# best_val_loss = float("inf")
# changed version:

best_val_macro_f1 = -1.0
history = []

for epoch in range(EPOCHS):
    train_loss, train_acc, train_macro_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, val_macro_f1 = validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step(val_loss)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
        f"Val Macro-F1: {val_macro_f1:.4f}"
    )

    # # Save best model based on validation loss
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     print("Saved best model")
    # changed to be consistent to the densenet model (macro f1 score)
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_macro_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"),
        index=False,
    )

    # Save best model based on validation macro-F1
    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "num_classes": NUM_CLASSES,
            "architecture": "resnet101",
            # changed for weighted model:
            # "loss": "CrossEntropyLoss",
            "loss": "WeightedCrossEntropyLoss",
            "weighted_loss": True,
            "class_counts": class_counts.to_dict(),
            "class_weights": class_weights.detach().cpu().tolist(),
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "seed": 42,
        }

        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"Saved best model with val_macro_f1={best_val_macro_f1:.4f}")

def test_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    # added zero warning 
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro",zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro",zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro",zero_division=0)

    print("\nTest Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


    # file outputs
    test_results = pd.DataFrame([{
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }])

    test_results.to_csv(
        os.path.join(OUTPUT_DIR, "test_results.csv"),
        index=False,
    )

    return accuracy, precision, recall, f1


# # Load best checkpoint
# model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))


checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

test_model(model, test_loader)