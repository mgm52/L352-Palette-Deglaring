import argparse
from data.flare7k_dataset import Flare_Image_Loader
import os
import glob
import code
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-pre', '--prefix', type=str, default="0")
    args = parser.parse_args()

    prefix = args.prefix

    IMG_SIZE = 256

    SAVE = True
    OVERWRITE = False
    save_dir = f"datasets/presynth_flare7k_{IMG_SIZE}"

    if SAVE:
        if os.path.exists(save_dir) and OVERWRITE:
            input(f"DELETING {save_dir}: ARE YOU SURE?")
            print(f"Deleting {save_dir}...")
            os.system(f"rm -rf {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, "gt"))
            os.makedirs(os.path.join(save_dir, "flare_added"))
            os.makedirs(os.path.join(save_dir, "flare"))
            os.makedirs(os.path.join(save_dir, "mask"))

    ds = Flare_Image_Loader(
        bg_path=          "datasets/Flickr24K/raw",
        flare_path=       "datasets/flare/Flare",
        lsource_path=     "datasets/flare/Annotations/Light_Source",
        transform_base=   {
                                "pre_crop_size": None,
                                "img_size": IMG_SIZE,
                            },
        transform_flare=  {
                                "scale_min": 0.03 * IMG_SIZE/64,   # default 0.8
                                "scale_max": 0.15 * IMG_SIZE/64,   # default 1.5
                                "shear": 20         # chooses a random angle (deg) between -shear and +shear
                            },
        mask_type=        "luminance",
        mask_high_on_lsource= False,
        placement_mode=   "random",                # "light_pos", "random", or "centre"
        num_sources= [1.5, 1.5]
    )

    epoch = 2
    while True:
        for i in tqdm(range(len(ds))):
            ex1 = ds[i]
            # This is a dict with keys "gt_image", "flare", "cond_image", "gamma".
            # Visualize gt_image, flare, lq on a plot and show it:
            # Note: straight up using .imshow would produce Invalid shape (3, 256, 256).
            # Instead: transpose the shape to (256, 256, 3).

            if SAVE:
                # Save the images to save_dir
                save_image(ex1["gt_image"], os.path.join(save_dir, f"gt/gt_img_{prefix}_{epoch}_{i}.png"))
                save_image(ex1["cond_image"], os.path.join(save_dir, f"flare_added/flare_added_img_{prefix}_{epoch}_{i}.png"))
                save_image(ex1["flare"], os.path.join(save_dir, f"flare/flare_img_{prefix}_{epoch}_{i}.png"))
                save_image(ex1["mask"], os.path.join(save_dir, f"mask/mask_img_{prefix}_{epoch}_{i}.png"))
            else:
                imgs = ["gt_image", "flare", "cond_image"]
                optionals = ["lsource", "mask", "lsource_mask", "flare_mask"]
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


                plt.suptitle(f"Img {i}: gamma {ex1['gamma']}")
                plt.show()
        epoch += 1
            
