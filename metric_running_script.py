

import argparse
from PIL import Image
import models.metric as met
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="C:\\Users\\Max\\Downloads\\out_def.png")
    parser.add_argument('-t', '--target', type=str, default="C:\\Users\\Max\\Downloads\\gt_def.png")
    args = parser.parse_args()

    input = args.input
    target = args.target

    input_img = Image.open(input).convert('RGB')
    target_img = Image.open(target).convert('RGB')

    to_tensor=transforms.ToTensor()
    input_img = to_tensor(input_img)
    target_img = to_tensor(target_img)

    print(target_img)
    print(input_img)

    psnr = met.psnr(input_img, target_img)
    psnr_y = met.psnr_y(input_img, target_img)
    mae = met.mae(input_img, target_img)
    mse = met.mse(input_img, target_img)
    ssim = met.ssim(input_img, target_img)
    ssim_y = met.ssim_y(input_img, target_img)

    metrics = {"psnr": psnr, "psnr_y": psnr_y, "mae": mae, "mse": mse, "ssim": ssim, "ssim_y": ssim_y}
    print(metrics)