import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    dir = './datasets/Flickr24K/colorpng'
    # Rename every file from n.jpg to nnnnn.jpg
    for i, file in enumerate(tqdm(os.listdir(dir))):
        os.rename(os.path.join(dir, file), os.path.join(dir, '{:05d}.png'.format(i)))