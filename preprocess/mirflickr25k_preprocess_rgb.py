import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

if __name__ == '__main__':
    home = './' # path saved .npy
    flist_save_path = './flist'
    image_save_path = './images' # images save path
    
    # Convert every color image into grayscale
    color_save_path, gray_save_path  = '{}/color'.format(image_save_path), '{}/gray'.format(image_save_path)
    os.makedirs(gray_save_path, exist_ok=True)
    
    # Load coloured images from color_save_path, each being .jpg
    imgs = os.listdir(color_save_path)
    print("Loaded listdir")
    imgs = [os.path.join(color_save_path, img) for img in imgs]
    imgs = [cv2.imread(img) for img in imgs]
    print("Finished imread images")

    # Convert to grayscale
    imgsg = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    print("Finished converting to grayscale")
    imgsg = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in imgsg]
    print("Finished converting to rgb")

    # Save grayscale images to gray_save_path, each being .png
    for i, img in enumerate(imgsg):
        cv2.imwrite(os.path.join(gray_save_path, '{:05d}.png'.format(i)), img)
    print("Finished saving grayscale images")
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(color_save_path, '{:05d}.png'.format(i)), img)
    print("Finished saving colour images")

    os.makedirs(flist_save_path, exist_ok=True)
    arr = np.random.permutation(25000)
    with open('{}/train.flist'.format(flist_save_path), 'w') as f:
        for item in arr[:24000]:
            print(str(item).zfill(5), file=f)
    with open('{}/test.flist'.format(flist_save_path), 'w') as f:
        for item in arr[24000:]:
            print(str(item).zfill(5), file=f)