import os
from torch.utils.data import Dataset
from PIL import Image

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, f) for f in os.listdir(root)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.splitext(os.path.basename(path))[0]
        return image, filename

