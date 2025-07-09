# File: src/transforms.py
"""
Image transform functions for SSL and DINO.
"""
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def med_transform_ssl(img_size=224, gray=True):
    """
    Grayscale Resize + Normalize transform for SSL pretraining.
    """
    mean, std = ([0.5], [0.5]) if gray else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class DinoTransform:
    """
    Generate two global crop views for DINO.
    """
    def __init__(self, img_size=224):
        flip = transforms.RandomHorizontalFlip()
        color = transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
        )
        gray = transforms.RandomGrayscale(p=0.2)
        normalize = transforms.Normalize([0.5], [0.5])

        self.global_t = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            flip, color, gray,
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        g1 = self.global_t(x)
        g2 = self.global_t(x)
        return g1.repeat(3, 1, 1), g2.repeat(3, 1, 1)
