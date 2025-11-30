import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from models import ResUnet , Unet , PSPNet
from datasets.figaro import FigaroDataset
from utils.metrics import dice_score, iou_score
from utils.visualization import save_prediction
from utils.transforms import img_transforms , mask_transforms
from datasets.celeb import celeb_Dataset


def load_model(model_name, checkpoint_path, device):
    if model_name == "ResUnet":
        model = ResUnet(in_channels=3, n_classes=1)
    elif model_name == "Unet":
        model = Unet(in_channels=3, out_channels=1)
    elif model_name == "PSPNet":
        model = PSPNet(in_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, criterion, device):
    total_loss = 0
    total_dice = 0
    total_iou = 0
    n = len(dataloader.dataset)

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            total_loss += loss.item() * imgs.size(0)
            total_dice += dice_score(preds, masks) * imgs.size(0)
            total_iou += iou_score(preds, masks) * imgs.size(0)

    return total_loss / n, total_dice / n, total_iou / n


def save_examples(model, dataloader, save_dir, device, max_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    shuffled_loader = DataLoader(
        dataloader.dataset, 
        batch_size=dataloader.batch_size, 
        shuffle=True 
    )
    count = 0

    with torch.no_grad():
        for imgs, masks in shuffled_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            save_path = os.path.join(save_dir, f"sample_{count}")
            save_prediction(imgs[0].cpu(), masks[0].cpu(), preds[0].cpu(), save_path)

            count += 1
            if count >= max_samples:
                break


def main():
    model_name = "ResUnet"  # change to "Unet" or "PSPNet"
    exp_dir = f"experiments/{model_name}"
    checkpoint = f"{exp_dir}/{model_name}_best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    figaro_test = FigaroDataset(
        root_dir="./data/raw",
        split="Testing",
        transform=img_transforms,
        target_transform=mask_transforms
    )

    celeb_test = celeb_Dataset(
        root_dir="./data/raw",
        split="testing",  
        transform=img_transforms,
        target_transform=mask_transforms
    )

    combined_test_dataset = ConcatDataset([figaro_test, celeb_test])


    test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_name, checkpoint, device)
    criterion = nn.BCELoss()

    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)

    print("===== Evaluation Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU : {test_iou:.4f}")

    with open(f"{exp_dir}/eval_report.txt", "w") as f:
        f.write("Evaluation Report\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")
        f.write(f"Test IoU : {test_iou:.4f}\n")

    save_examples(model, test_loader, f"{exp_dir}/example_preds", device)
    print("Saved example predictions.")


if __name__ == "__main__":
    main()
