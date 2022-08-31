import numpy as np
import torch
from model import load_model, FeatureExtractor
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from PIL import Image
from torchvision import transforms
import time

CLASS_NAME = "lens"
MODEL_NAME = "lens_B5_1_1000_n-512"
IMG_SIZE = (512, 512)
DEVICE = 'cuda'
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PRE_EXTRACTED = False
EXTRACTOR = "effnetB5"
N_FEAT = {"effnetB5": 304}[EXTRACTOR]
THRESHOLD = 0.6938925385475159

localize = True
upscale_mode = 'bilinear'

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]

def viz_maps(img_path, z, name):
    map_export_dir = join('/home/synapse/simulator/backend/static/test_map', "")
    os.makedirs(map_export_dir, exist_ok=True)

    image = PIL.Image.open(img_path).convert('RGB')
    image = np.array(image)

    z_grouped = list()
    likelihood_grouped = list()
    all_maps = list()
    for i in range(len(z)):
        #print(len(z))
        z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
        likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))    

    all_maps.extend(likelihood_grouped[0])
    map_to_viz = t2np(F.interpolate(all_maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
        0, 0]

    plt.clf()
    plt.imshow(map_to_viz)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)

    plt.clf()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
    plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
    plt.savefig(join(map_export_dir, 'overlay.jpg'), bbox_inches='tight', pad_inches=0)
    return

def evaluate_one(model, img_path):
    model.to(DEVICE)
    model.eval()

    if not PRE_EXTRACTED:
        fe = FeatureExtractor()
        fe.eval()
        fe.to(DEVICE)
        for param in fe.parameters():
            param.requires_grad = False

    img = Image.open(img_path)
    transform = transforms.Resize(IMG_SIZE)
    start = time.time()
    
    tfs = [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    transform = transforms.Compose(tfs)

    img = transform(img)

    # preprocess_batch
    '''move data to device and reshape image'''
    img = img.to(DEVICE)
    img = img.view(-1, *img.shape[-3:])
    
    
    if not PRE_EXTRACTED:
        img = fe(img)
    z = model(img)

    print(model)

    z_concat = t2np(concat_maps(z))

    nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
    end = time.time()

    print(end - start)

    viz_maps(img_path, z, "test_img")

    print(nll_score)
    return {
        "anomaly_score" : nll_score,
        "isAnomaly" : nll_score <= THRESHOLD,
    }

def evaluate_function():
    mod = load_model(MODEL_NAME)
    evaluate_one(mod, "test_img.jpg")

evaluate_function()
