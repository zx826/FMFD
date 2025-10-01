# FMFD
# Requirments
python==3.10

pytorch==2.1.1

mamba-ssm==2.2.2

causal-convld==1.4.0

timm==1.0.7

This project is best run on Linux.
# TEST
All images need to be resized to a fixed size (e.g., 256×256 or 512×512). In our paper, 256×256 was used. The 256×256 pretrained weights can be downloaded from Baidu Netdisk: 

Link: https://pan.baidu.com/s/1OMMeZDFxY8V3drw-9GYzlw 

Extraction code: gq4x

    python distillation_test.py --data-path ./data --out-path ./generate --checkpoint-path ./checkpoint --image-size 256

Due to computational limitations, I made some modifications to the original student model at 512 resolution by introducing wavelets for compression. The practical results are largely similar and are provided for reference only.

The 512×512 pretrained weights can be downloaded from Baidu Netdisk:

Link: https://pan.baidu.com/s/14arXb51I3qS4FOIYkWoisA

Extraction code: 8q6b

    python distillation_test.py --data-path ./data --out-path ./generate --checkpoint-path ./checkpoint --image-size 512

# TRAIN
The training code is not publicly available at this time.
