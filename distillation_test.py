import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, ConcatDataset
from Dataset import ImageFolder
from torch.utils.data import SequentialSampler
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from glob import glob
from time import time
import argparse
import os
from distillation_model import UMamba,UMamba_512, MambaConfig
from pytorch_wavelets import DWTInverse,DWTForward


def requires_grad(model, flag=True):  
    for p in model.parameters():
        p.requires_grad = flag

def center_crop_arr(pil_image, image_size):  

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def load_checkpoint(checkpoint_path, model):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"]) 
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return False


def main(args):  

    assert torch.cuda.is_available(), "At least one GPU is required"
    print('GPU：', torch.cuda.is_available())

 
    if args.image_size == 256:
        modelfig = MambaConfig(d_model=1024, n_layers=4)  
        model = UMamba(modelfig)
        model = model.to('cuda')
        print(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        checkpoint_dir = os.path.join(args.checkpoint_path,"256")
    elif args.image_size == 512:
        modelfig = MambaConfig(d_model=1024, n_layers=4)  
        model = UMamba_512(modelfig)
        model = model.to('cuda')
        dwt = DWTForward(J=1, wave='bior3.5', mode='periodization').to('cuda')
        idwt = DWTInverse(wave='bior3.5', mode='periodization').to('cuda')
        print(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        checkpoint_dir = os.path.join(args.checkpoint_path,"512")
    else:
        print(f"Unsupported image size: {args.image_size}. Please use 256 or 512.")

 
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    data_raw = ImageFolder(args.data_path, transform=transform) 
    raw = SequentialSampler(data_raw)
    loader_raw = DataLoader(
        data_raw,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=raw,
        pin_memory=True,
        drop_last=False
    )
                 
    model.eval()

    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoint_files:
            last_checkpoint_path = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_path)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None

    if checkpoint_path:
        success = load_checkpoint(checkpoint_path, model)
    else:
        success = False
    if success:
        print("Checkpoint loaded successfully. Continuing training...")
    else:
        print("No checkpoint found. retraining please...")
   
 
    if args.image_size == 256:
        for u,names in loader_raw:
            u = u.to('cuda')
            name = "_".join(names)
            with torch.no_grad():
                 start_time = time()
                 _,_,_,_,out = model(u)
                 end_time = time()
                 t = end_time - start_time
                 print(f"{name}.png,time: {t:.4f}s")                      
            save_image(out, os.path.join(args.out_path, f"{name}.png"), nrow=4, normalize=True, value_range=(-1, 1))
    
    elif args.image_size == 512:
        for u,names in loader_raw:
            u = u.to('cuda')
            name = "_".join(names)
            with torch.no_grad():
                 start_time = time()
                 u0,H = dwt(u)
                 h1,x32,x33,x_3,x_2,out = model(u0,H[0])
                 h11, h12, h13 = torch.chunk(h1, 3, dim=0)
                 highs = torch.stack([h11, h12, h13], dim=2)  # (B,C,3,H,W)
                 out = idwt((out, [highs]))
                 end_time = time() 
                 t = end_time - start_time
                 print(f"{name}.png,time: {t:.4f}s")                           
            save_image(out, os.path.join(args.out_path, f"{name}.png"), nrow=4, normalize=True, value_range=(-1, 1))
     
           






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--out-path", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default="")
    parser.add_argument("--global-batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
