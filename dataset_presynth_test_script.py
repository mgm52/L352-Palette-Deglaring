from data.presynth_flare7k_dataset import Flare_Image_Loader_Presynth
import os
import glob
import code
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

if __name__ == "__main__":


    ds = Flare_Image_Loader_Presynth(
        "datasets/presynth_flare7k_64",
        mask_high_on_lsource=False,
        mask_type="luminance",
        mask_gamma=10.0)

    epoch = 2
    while True:
        for i in tqdm(range(len(ds))):
            ex1 = ds[i]
            # This is a dict with keys "gt_image", "flare", "cond_image", "mask".
            # Visualize gt_image, flare, lq on a plot and show it:
            # Note: straight up using .imshow would produce Invalid shape (3, 256, 256).
            # Instead: transpose the shape to (256, 256, 3).

            imgs = ["gt_image", "flare", "cond_image"]
            optionals = ["mask"]
            for opt in optionals:
                if ex1[opt] is not None: imgs.append(opt)

            # TODO: check whether we can derive gt_image by subtracting flare
            fig, axs = plt.subplots(1, len(imgs)+1)
            for j, img in enumerate(imgs):
                axs[j].imshow(ex1[img].permute(1, 2, 0))
                # Remove ticks
                axs[j].set_xticks([])
                axs[j].set_yticks([])
                # Set title
                axs[j].set_title(f"{img}: max {ex1[img].max():.2f}, min {ex1[img].min():.2f}")
            
            subtracted = ex1["cond_image"] - ex1["gt_image"]
            axs[-1].imshow(subtracted.permute(1, 2, 0))
            axs[-1].set_xticks([])
            axs[-1].set_yticks([])
            axs[-1].set_title(f"subtracted: max {subtracted.max():.2f}, min {subtracted.min():.2f}")


            plt.suptitle(f"Img {i}: path {ex1['path']}")
            plt.show()
        epoch += 1
            
