import code
from PIL import Image
import numpy as np
from functools import partial
import os
from os import path as osp
# from crowdai_utils import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from config import CFG

import torch
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn.utils.rnn import pad_sequence


class InriaCocoDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, tokenizer=None, shuffle_tokens=False):
        image_dir = osp.join(dataset_dir, "images")
        self.image_dir = image_dir
        self.annotations_path = osp.join(dataset_dir, "annotation.json")
        self.transform = transform
        self.tokenizer = tokenizer
        self.shuffle_tokens = shuffle_tokens
        # self.images = os.listdir(self.image_dir)
        self.coco = COCO(self.annotations_path)
        # self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.images = [file for file in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, file))]
        self.image_ids = [int(im.split('-')[-1].split('.')[0]) for im in self.images if im.split('-')[0] not in ['kitsap4', 'kitsap5']]

    def __len__(self):
        return len(self.image_ids)

    def annToMask(self):
        return

    def shuffle_perm_matrix_by_indices(self, old_perm: torch.Tensor, shuffle_idxs: np.ndarray):
        Nv = old_perm.shape[0]
        padd_idxs = np.arange(len(shuffle_idxs), Nv)
        shuffle_idxs = np.concatenate([shuffle_idxs, padd_idxs], axis=0)

        transform_arr = torch.zeros_like(old_perm)
        for i in range(len(shuffle_idxs)):
            transform_arr[i, shuffle_idxs[i]] = 1.

        # https://math.stackexchange.com/questions/2481213/adjacency-matrix-and-changing-order-of-vertices
        new_perm = torch.mm(torch.mm(transform_arr, old_perm), transform_arr.T)
        # new_perm = torch.zeros_like(old_perm)

        # # generate new perm matrix based on shuffling indices.
        # for i in range(len(shuffle_idxs)):
        #     new_i = shuffle_idxs[i]
        #     new_j = shuffle_idxs[old_perm[i].nonzero().item()]
        #     new_perm[new_i, new_j] = 1.

        # # Add self connections of unconnected vertices.
        # for i in range(Nv):
        #     row = new_perm[i, :]
        #     col = new_perm[:, i]
        #     if torch.sum(row) == 0 or torch.sum(col) == 0:
        #         new_perm[i, i] = 1.

        return new_perm

    def __getitem__(self, index):
        n_vertices = CFG.N_VERTICES
        img_id = self.image_ids[index]
        img = self.coco.loadImgs(img_id)[0]
        img_path = osp.join(self.image_dir, img["file_name"])
        ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(ann_ids)  # annotations of all instances in an image.

        image = np.array(Image.open(img_path).convert("RGB"))

        mask = np.zeros((img['width'], img['height']))
        corner_coords = []
        corner_mask = np.zeros((img['width'], img['height']), dtype=np.float32)
        perm_matrix = np.zeros((n_vertices, n_vertices), dtype=np.float32)
        for ins in annotations:
            segmentations = ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                segm[:, 0] = np.clip(segm[:, 0], 0, img['width'] - 1)
                segm[:, 1] = np.clip(segm[:, 1], 0, img['height'] - 1)
                points = segm[:-1]
                corner_coords.extend(points.tolist())
                mask += self.coco.annToMask(ins)
        mask = mask / 255. if mask.max() == 255 else mask
        mask = np.clip(mask, 0, 1)

        # corner_coords = np.clip(np.array(corner_coords), 0, 299)
        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)

        if len(corner_coords) > 0.:
            corner_mask[corner_coords[:, 0], corner_coords[:, 1]] = 1.
        # corner_coords = (corner_coords / img['width']) * CFG.INPUT_WIDTH

        ############# START: Generate gt permutation matrix. #############
        v_count = 0
        for ins in annotations:
            segmentations = ins['segmentation']
            for idx, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                points = segm[:-1]
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    if v_count+i > n_vertices - 1 or v_count+j > n_vertices-1:
                        break
                    perm_matrix[v_count+i, v_count+j] = 1.
                v_count += len(points)

        for i in range(v_count, n_vertices):
            perm_matrix[i, i] = 1.

        # Workaround for open contours:
        for i in range(n_vertices):
            row = perm_matrix[i, :]
            col = perm_matrix[:, i]
            if np.sum(row) == 0 or np.sum(col) == 0:
                perm_matrix[i, i] = 1.
        perm_matrix = torch.from_numpy(perm_matrix)
        ############# END: Generate gt permutation matrix. #############

        masks = [mask, corner_mask]

        if len(corner_coords) > CFG.N_VERTICES:
            corner_coords = corner_coords[:CFG.N_VERTICES]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks, keypoints=corner_coords.tolist())
            image = augmentations['image']
            mask = augmentations['masks'][0]
            corner_mask = augmentations['masks'][1]
            corner_coords = np.array(augmentations['keypoints'])

        if self.tokenizer is not None:
            coords_seqs, rand_idxs = self.tokenizer(corner_coords, shuffle=self.shuffle_tokens)
            coords_seqs = torch.LongTensor(coords_seqs)
            # perm_matrix = torch.cat((perm_matrix[rand_idxs], perm_matrix[len(rand_idxs):]))
            if self.shuffle_tokens:
                perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)
        else:
            coords_seqs = corner_coords

        return image, mask[None, ...], corner_mask[None, ...], coords_seqs, perm_matrix


class InriaCocoDataset_val(Dataset):
    def __init__(self, cfg, dataset_dir, transform=None, tokenizer=None, shuffle_tokens=False):
        self.CFG = cfg
        image_dir = osp.join(dataset_dir, "images")
        self.image_dir = image_dir
        self.annotations_path = osp.join(dataset_dir, "annotation.json")
        self.transform = transform
        self.tokenizer = tokenizer
        self.shuffle_tokens = shuffle_tokens
        # self.images = os.listdir(self.image_dir)
        self.coco = COCO(self.annotations_path)
        # self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())
        self.images = [file for file in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, file))]
        self.image_ids = [int(im.split('-')[-1].split('.')[0]) for im in self.images]

    def __len__(self):
        return len(self.image_ids)

    def annToMask(self):
        return

    def shuffle_perm_matrix_by_indices(self, old_perm: torch.Tensor, shuffle_idxs: np.ndarray):
        Nv = old_perm.shape[0]
        padd_idxs = np.arange(len(shuffle_idxs), Nv)
        shuffle_idxs = np.concatenate([shuffle_idxs, padd_idxs], axis=0)

        transform_arr = torch.zeros_like(old_perm)
        for i in range(len(shuffle_idxs)):
            transform_arr[i, shuffle_idxs[i]] = 1.

        # https://math.stackexchange.com/questions/2481213/adjacency-matrix-and-changing-order-of-vertices
        new_perm = torch.mm(torch.mm(transform_arr, old_perm), transform_arr.T)

        return new_perm

    def __getitem__(self, index):
        n_vertices = self.CFG.N_VERTICES
        img_id = self.image_ids[index]
        img = self.coco.loadImgs(img_id)[0]
        img_path = osp.join(self.image_dir, img["file_name"])
        ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(ann_ids)  # annotations of all instances in an image.

        image = np.array(Image.open(img_path).convert("RGB"))

        mask = np.zeros((img['width'], img['height']))
        corner_coords = []
        corner_mask = np.zeros((img['width'], img['height']), dtype=np.float32)
        perm_matrix = np.zeros((n_vertices, n_vertices), dtype=np.float32)
        for ins in annotations:
            segmentations = ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                segm[:, 0] = np.clip(segm[:, 0], 0, img['width'] - 1)
                segm[:, 1] = np.clip(segm[:, 1], 0, img['height'] - 1)
                points = segm[:-1]
                corner_coords.extend(points.tolist())
                mask += self.coco.annToMask(ins)
        mask = mask / 255. if mask.max() == 255 else mask
        mask = np.clip(mask, 0, 1)

        # corner_coords = np.clip(np.array(corner_coords), 0, 299)
        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)

        if len(corner_coords) > 0.:
            corner_mask[corner_coords[:, 0], corner_coords[:, 1]] = 1.
        # corner_coords = (corner_coords / img['width']) * CFG.INPUT_WIDTH

        ############# START: Generate gt permutation matrix. #############
        v_count = 0
        for ins in annotations:
            segmentations = ins['segmentation']
            for idx, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)
                points = segm[:-1]
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    if v_count+i > n_vertices - 1 or v_count+j > n_vertices-1:
                        break
                    perm_matrix[v_count+i, v_count+j] = 1.
                v_count += len(points)

        for i in range(v_count, n_vertices):
            perm_matrix[i, i] = 1.

        # Workaround for open contours:
        for i in range(n_vertices):
            row = perm_matrix[i, :]
            col = perm_matrix[:, i]
            if np.sum(row) == 0 or np.sum(col) == 0:
                perm_matrix[i, i] = 1.
        perm_matrix = torch.from_numpy(perm_matrix)
        ############# END: Generate gt permutation matrix. #############

        masks = [mask, corner_mask]

        if len(corner_coords) > self.CFG.N_VERTICES:
            corner_coords = corner_coords[:self.CFG.N_VERTICES]

        if self.transform is not None:
            augmentations = self.transform(image=image, masks=masks, keypoints=corner_coords.tolist())
            image = augmentations['image']
            mask = augmentations['masks'][0]
            corner_mask = augmentations['masks'][1]
            corner_coords = np.array(augmentations['keypoints'])

        if self.tokenizer is not None:
            coords_seqs, rand_idxs = self.tokenizer(corner_coords, shuffle=self.shuffle_tokens)
            coords_seqs = torch.LongTensor(coords_seqs)
            # perm_matrix = torch.cat((perm_matrix[rand_idxs], perm_matrix[len(rand_idxs):]))
            if self.shuffle_tokens:
                perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)
        else:
            coords_seqs = corner_coords

        return image, mask[None, ...], corner_mask[None, ...], coords_seqs, perm_matrix, torch.tensor([img['id']])


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch = [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)

    coords_seq_batch = pad_sequence(
        coords_seq_batch,
        padding_value=pad_idx,
        batch_first=True
    )

    if max_len:
        pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

    image_batch = torch.stack(image_batch)
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch


class InriaCocoDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, file))]
    
    def __getitem__(self, index):
        img_path = osp.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)['image']

        image = torch.FloatTensor(image)
        return image
    
    def __len__(self):
        return len(self.images)