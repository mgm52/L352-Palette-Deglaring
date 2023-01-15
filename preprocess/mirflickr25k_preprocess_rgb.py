import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    #home = './' # path saved .npy
    flist_save_path = './datasets/Flickr24K/flist'
    image_save_path = './datasets/Flickr24K' # images save path
    image_load_path = './datasets/Flickr24K/raw'
    
    # Convert every color image into grayscale
    color_save_path, gray_save_path  = '{}/colorpng'.format(image_save_path), '{}/graypng'.format(image_save_path)
    os.makedirs(gray_save_path, exist_ok=True)
    
    # Load coloured images from origin path, each being .jpg
    imgs = os.listdir(image_load_path)
    print("Loaded listdir")
    imgs = [os.path.join(image_load_path, img) for img in imgs]
    # use tqdm to imread each image
    imgs = [cv2.imread(img) for img in tqdm(imgs)]
    print("Finished imread images")

    # Convert to grayscale
    imgsg = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    print("Finished converting to grayscale")
    imgsg = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in imgsg]
    print("Finished converting to rgb")

    # Save grayscale images to gray_save_path, each being .png
    for i, img in enumerate(tqdm(imgsg)):
        cv2.imwrite(os.path.join(gray_save_path, '{:05d}.png'.format(i)), img)
    print("Finished saving grayscale images")
    for i, img in enumerate(tqdm(imgs)):
        cv2.imwrite(os.path.join(color_save_path, '{:05d}.png'.format(i)), img)
    print("Finished saving colour images")

    os.makedirs(flist_save_path, exist_ok=True)
    arr = np.random.permutation(len(imgs))
    with open('{}/train.flist'.format(flist_save_path), 'w') as f:
        for item in arr[:len(imgs)-1000]:
            print(str(item).zfill(5), file=f)
    with open('{}/test.flist'.format(flist_save_path), 'w') as f:
        for item in arr[len(imgs)-1000:]:
            print(str(item).zfill(5), file=f)
    print("Finished all.")