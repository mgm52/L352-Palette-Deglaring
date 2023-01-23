import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def mse(input, target):
    with torch.no_grad():
        loss = nn.MSELoss()
        output = loss(input, target)
    return output

def psnr(input, target):
    with torch.no_grad():
        mse = nn.MSELoss()
        output = 10 * torch.log10(1 / mse(input, target))
    return output

def psnr_y(input, target):
    input_y, target_y = to_y_channel(input), to_y_channel(target)
    return psnr(input_y, target_y)

def ssim(input, target):
    from skimage.metrics import structural_similarity

    assert len(input.shape) in [2, 3] and len(target.shape) in [2, 3], f"Input must be a 3D tensor, but given {input.shape} and {target.shape}"

    if input.shape[0] == 3:
        input = input.permute(1, 2, 0)
    if target.shape[0] == 3:
        target = target.permute(1, 2, 0)

    # Use structural_similarity to compute ssim
    ssim = structural_similarity(input.numpy(), target.numpy(), multichannel=True)
    return ssim

def ssim_y(input, target):
    # Convert input and target to Y channel of YCbCr
    input_y, target_y = to_y_channel(input), to_y_channel(target)
    return ssim(input_y, target_y)

def to_y_channel(img):
    if len(img.shape) == 3:
        return 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    else:
        return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)