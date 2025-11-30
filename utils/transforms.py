from torchvision import transforms
import torchvision.transforms.functional as TF
import random
img_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

mask_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])



def simple_augment(image, mask):
    """
    image: PIL Image (RGB)
    mask: PIL Image (grayscale)
    Returns: (image_tensor, mask_tensor)
    """
    # Flip both together (50% chance)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    
    # Rotate both together (50% chance, same angle)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
    
    # Resize both to 256x256
    image = TF.resize(image, (256, 256))
    mask = TF.resize(mask, (256, 256))
    
    # Convert both to tensors
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)
    
    return image, mask