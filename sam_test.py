from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
import time
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        img[m] = color_mask

    return img

# on brief examination runtimes are similar enough and H model
# seems higher quality
SAM_H = ("vit_h","sam_vit_h_4b8939.pth") # 2-2.5s and 6.8GB
SAM_L = ("vit_l","sam_vit_l_0b3195.pth") # 2-2.5s and 5.3GB
SAM_B = ("vit_b","sam_vit_b_01ec64.pth") # 1.5s and 4.4GB

(model,path) = SAM_H
sam = sam_model_registry[model](checkpoint=f"../shared/sam/{path}")
sam.to(device=0)
mask_generator = SamAutomaticMaskGenerator(sam)

# SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )


img_path = "atari-benchmarks/frostbite_1/674.png"

with Image.open(img_path) as im:
    img = np.asarray(im)


print("Generating masks...")

start = time.time()
masks = mask_generator.generate(img)
end = time.time()
print(f"Generated for {img.shape} in {end - start} seconds")
mask_img = show_anns(masks)

res = np.concatenate([img/255, mask_img], axis=1)

import os
os.makedirs("out/sam/", exist_ok=True)
Image.fromarray((res*255).astype("uint8")).save(f"out/sam/{model}.png")

# breakpoint()