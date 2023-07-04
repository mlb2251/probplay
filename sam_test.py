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
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        img[m] = color_mask
    # ax.imshow(img)

    return img

# plt.figure(figsize=(5,5))
# plt.imshow(img)
# plt.axis('off')
# plt.show()


sam = sam_model_registry["default"](checkpoint="/Users/matthewbowers/proj/shared/sam/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

with Image.open("atari-benchmarks/frostbite_1/674.png") as im:
    img = np.asarray(im)


print("Generating masks...")

start = time.time()
masks = mask_generator.generate(img)
end = time.time()
print(f"Generated for {img.shape} in {end - start} seconds")
mask_img = show_anns(masks)

res = np.concatenate([img/255, mask_img], axis=1)


Image.fromarray((res*255).astype("uint8")).save("out.png")

# plt.figure(figsize=(5,5))
# plt.imshow(res)
# plt.axis('off')
# plt.show()


# plt.show()
# breakpoint()