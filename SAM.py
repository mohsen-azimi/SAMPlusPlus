"""Segment_anything.ipynb    https://colab.research.google.com/drive/1VeU6OWmrWylZ3568BsUXGlO8N9Fyv0CQ
 pip install 'git+https://github.com/facebookresearch/segment-anything.git'
"""
import torch  # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
from git_update import git_add_commit_push
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

git_add_commit_push()

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# file_name = "dog"
# file_name = '20230412_103006' # leila's test
file_name = 'bridge'

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


image = cv2.imread(f'images/{file_name}.jpg')  # Copy the link of your image
# resize if needed (by percentage)
# image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

sys.path.append("..")

sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# select device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)
###########################################################
print(len(masks))
print(masks[0].keys())

print(masks[0]['segmentation'])

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
###########################################################
# plot masks in a grid of 3x3
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(masks[i]['segmentation'])
    plt.axis('off')
plt.show()
##########################################################
# save all the masks in a folder named "masks/file_name"
# check if the folder exists
if not os.path.exists(f'masks/{file_name}'):
    os.makedirs(f'masks/{file_name}')
for i in range(len(masks)):
    mask = masks[i]['segmentation']
    mask = mask * 255
    mask = mask.astype(np.uint8)
    cv2.imwrite(f'masks/{file_name}/{file_name}_{i}.png', mask)

