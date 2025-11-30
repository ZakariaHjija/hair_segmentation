import os
from PIL import Image
from torch.utils.data import Dataset


class celeb_Dataset(Dataset):
    def __init__(self, root_dir, split="training", transform=None, target_transform=None,augment_fn=None):
        self.img_dir = os.path.join(root_dir, "CelebA_lite", split,"img")
        self.mask_dir = os.path.join(root_dir, "CelebA_lite", split,"masks")

        self.image_names = sorted(os.listdir(self.img_dir))
        self.mask_names = sorted(os.listdir(self.mask_dir))

        self.transform = transform
        self.target_transform = target_transform
        self.augment_fn = augment_fn
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment_fn:
            image, mask = self.augment_fn(image, mask)
        else:
            # Use separate transforms (for validation or no augmentation)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

        return image, mask
