import os
import time
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from functools import partial
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2

from test_config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    load_checkpoint,
    test_generate,
    postprocess,
    permutations_to_polygons,
)
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder
)

from torch.utils.data import DataLoader
from datasets.dataset_image_orientee import ImageOrienteeDatasetTest
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
import torch.multiprocessing
from shapely import Polygon
import shapely
import geopandas as gpd
torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset to use for evaluation.")
parser.add_argument("-c", "--checkpoint_name", help="Choice of checkpoint to evaluate in experiment.")
parser.add_argument("-o", "--output_dir", help="Name of output subdirectory to store part predictions.")
args = parser.parse_args()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATASET = f"{args.dataset}"
DATASET_DIR = args.dataset
# PART_DESC = "val_images"
PART_DESC = f"{args.output_dir}"

CHECKPOINT_PATH = args.checkpoint_name
BATCH_SIZE = 24


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = int(image_id)
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    image_batch = []
    i_batch = []
    j_batch = []
    for image in batch:
        c, n, m = image.shape
        for i in range(2050, n, 180):
            for j in range(1740, m, 180):
                if i+224<n and j+224<m:
                    extrait = image[:, i:i+224, j:j+224]
                    image_batch.append(extrait/256)
                    i_batch.append(i)
                    j_batch.append(j)
    #image_batch.append(batch[0][:,4172:4172+224, 12826:12826+224]/256)
    #i_batch.append(4172)
    #j_batch.append(12826)
    #image_batch = image_batch[:10]
    #i_batch = i_batch[:10]
    #j_batch = j_batch[:10] 

    image_batch = torch.stack(image_batch)
    i_batch = torch.Tensor(i_batch)
    j_batch = torch.Tensor(j_batch)
    
    return image_batch, i_batch, j_batch


def main():
    seed_everything(42)

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    val_ds = ImageOrienteeDatasetTest(
        DATASET_DIR
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        num_workers=2
    )

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)
    model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        speed = []
        for _, (tiles, i_batch, j_batch) in enumerate(val_loader):
            polygones = []
            
            for k in tqdm(range(tiles.shape[0])):
                
                x = tiles[k:k+1]
                all_coords = []
                all_confs = []
                t0 = time.time()
                
                batch_preds, batch_confs, perm_preds = test_generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
                speed.append(time.time() - t0)
                vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)
                
                all_coords.extend(vertex_coords)
                all_confs.extend(confs)

                coords = []
                for i in range(len(all_coords)):
                    if all_coords[i] is not None:
                        coord = torch.from_numpy(all_coords[i])
                    else:
                        coord = torch.tensor([])

                    padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                    coord = torch.cat([coord, padd], dim=0)
                    coords.append(coord)
                batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]               

                for pp in batch_polygons:
                    for p in pp:
                        coords = []
                        for i in range(p.shape[0]):
                            coords.append([ p[i,1], p[i,0]])
                        poly = Polygon(coords)
                        if poly.centroid.x >=0 and poly.centroid.x <= 202 and poly.centroid.y >=0 and poly.centroid.y <= 202:
                            poly_t = shapely.affinity.translate(poly, xoff=j_batch[k], yoff=i_batch[k])
                            poly_s = shapely.affinity.scale(poly_t, xfact=1, yfact=-1, origin=(0, 0))
                            polygones.append(poly_s)

            gpd.GeoDataFrame({"geometry":polygones}).to_file(os.path.join(args.output_dir, "prediction.gpkg"))





if __name__ == "__main__":
    main()

