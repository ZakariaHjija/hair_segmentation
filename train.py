import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , ConcatDataset
from models import ResUnet , Unet , PSPNet
from datasets.figaro import FigaroDataset
from datasets.celeb import celeb_Dataset
from utils import img_transforms, mask_transforms , dice_score 
import matplotlib.pyplot as plt
from utils.transforms import simple_augment
from utils.metrics import iou_score




def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_dice += dice_score(outputs, masks) * imgs.size(0)
        running_iou += iou_score(outputs, masks) * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_dice = running_dice / len(train_loader.dataset)
    epoch_iou = running_iou / len(train_loader.dataset)
    return epoch_loss, epoch_dice, epoch_iou

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * imgs.size(0)
            running_dice += dice_score(outputs, masks) * imgs.size(0)
            running_iou += iou_score(outputs, masks) * imgs.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_dice = running_dice / len(val_loader.dataset)
    val_iou = running_iou / len(val_loader.dataset)
    return val_loss, val_dice, val_iou

def get_model(model_name, in_channels=3, n_classes=1):
    if model_name.lower() == "unet":
        return Unet(in_channels=in_channels, n_classes=n_classes)
    elif model_name.lower() == "resunet":
        return ResUnet(in_channels=in_channels, n_classes=n_classes)
    elif model_name.lower() == "pspnet":
        return PSPNet(num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def main():

    DATASET_PATH_fig ="./data/raw"  # Path to Figaro1k dataset
    DATASET_PATH_celebA = "./data/processed"  # Path to CelebA_lite dataset
    augment_fn = None #simple_augment , simple_augment
    figaro_train_dataset = FigaroDataset(
        root_dir=DATASET_PATH_fig,
        split="Training",
        transform=img_transforms,
        target_transform=mask_transforms,
        augment_fn=augment_fn
    )
    
    figaro_val_dataset = FigaroDataset(
        root_dir=DATASET_PATH_fig,
        split="Testing",
        transform=img_transforms,
        target_transform=mask_transforms
    )

    celeb_train_dataset = celeb_Dataset(
        root_dir=DATASET_PATH_celebA,
        split="train",
        transform=img_transforms,
        target_transform=mask_transforms,
        augment_fn=augment_fn
    )

    celeb_val_dataset = celeb_Dataset(
        root_dir=DATASET_PATH_celebA,
        split="val",
        transform=img_transforms,
        target_transform=mask_transforms,
        
    )

    train_dataset = ConcatDataset([figaro_train_dataset, celeb_train_dataset])#celeb_train_dataset, figaro_train_dataset
    val_dataset = ConcatDataset([figaro_val_dataset,celeb_val_dataset]) #celeb_val_dataset, v
    
    

    model_name ="resunet" #"unet"   # ou "resunet" # "pspnet"
    exp_dir = f"experiments/{model_name}"
    os.makedirs(exp_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print

    batch_size = 16
    lr = 1e-3
    num_epochs = 80

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,      # essaie 4 ou 8
    pin_memory=True,    # optimise transfert CPU -> GPU
    persistent_workers=True
)

    val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

    model = get_model(model_name).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    best_dice = 0.0
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Dice: {val_dice:.4f} | Val   IoU: {val_iou:.4f}")

        if  val_iou > best_iou or val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            torch.save(model.state_dict(), f"{exp_dir}/{model_name}_best_model.pth")
            print("---------Saved new best model")

        
    epochs = range(1, num_epochs+1)

    # LOSS CURVE
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title(f"{model_name} Loss Curve")
    plt.savefig(f"{exp_dir}/loss_curve.png")
    plt.close()

    # DICE CURVE
    plt.figure()
    plt.plot(epochs, train_dices, label="Train Dice")
    plt.plot(epochs, val_dices, label="Val Dice")
    plt.legend()
    plt.title(f"{model_name} Dice Curve")
    plt.savefig(f"{exp_dir}/dice_curve.png")
    plt.close()

    # IO CURVE
    plt.figure()
    plt.plot(epochs, train_ious, label="Train IoU")
    plt.plot(epochs, val_ious, label="Val IoU")
    plt.legend()
    plt.title(f"{model_name} IoU Curve")
    plt.savefig(f"{exp_dir}/iou_curve.png")
    plt.close()

if __name__ == "__main__":
    main()