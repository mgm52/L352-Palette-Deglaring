from flare7k_dataset import Flare_Pair_Loader
import os
import glob
import code
import matplotlib.pyplot as plt

if __name__ == "__main__":
    bg_path = "datasets/Flickr24K"
    flare_path = "datasets/flare/Flare"
    lsource_path = "datasets/flare/Annotations/Light_Source"
    transform_base = {
        "img_size": 256 # up to 640
    } 
    transform_flare = {
        "scale_min": 0.8,
        "scale_max": 1.5,
        #"translate": 64,    # chooses a random translation (px) between -translate and +translate
        "shear": 20         # chooses a random angle (deg) between -shear and +shear
    }
    mask_type = "luminance"

    ds = Flare_Pair_Loader({
        "bg_path": bg_path,
        "flare_path": flare_path,
        "lsource_path": lsource_path,
        "transform_base": transform_base,
        "transform_flare": transform_flare,
        "mask_type": mask_type
    })


    for i in range(len(ds)):
        ex1 = ds[i]
        # This is a dict with keys "gt", "flare", "lq", "gamma".
        # Visualize gt, flare, lq on a plot and show it:
        # Note: straight up using .imshow would produce Invalid shape (3, 256, 256).
        # Instead: transpose the shape to (256, 256, 3).

        imgs = ["gt", "flare", "lq"]
        optionals = ["lsource", "mask", "lsource_mask", "combo_mask"]
        for opt in optionals:
            if ex1[opt] is not None: imgs.append(opt)

        fig, axs = plt.subplots(1, len(imgs))
        for j, img in enumerate(imgs):
            axs[j].imshow(ex1[img].permute(1, 2, 0))
            # Remove ticks
            axs[j].set_xticks([])
            axs[j].set_yticks([])
            # Set title
            axs[j].set_title(f"{img}: max {ex1[img].max():.2f}, min {ex1[img].min():.2f}")
        
        plt.suptitle(f"Img {i}: gamma {ex1['gamma']}")
        plt.show()
