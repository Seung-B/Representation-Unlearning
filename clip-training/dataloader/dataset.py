# dataloader here
from torch.utils.data import Dataset

import os
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from omegaconf import OmegaConf
import os.path as op
import random
import re

from utils import load_from_yaml_file, read_json, load_config_file

import pandas as pd
import torch
import numpy as np

#from coco_unlearn import *

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions


class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self, config, text_tokenizer, args, context_length=77, input_resolution=224):
        
        super(CLIP_COCO_dataset, self).__init__()

        self.config = config

        annotation_file = self.config.train_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)

        annotations = coco_unlearn_object(annotations, args.retrain)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = self.config.train_img_dir
        # print("img dir : ", self.img_dir)

        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length


    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)

        return img_input, text_input


class CLIP_CC3M_dataset(Dataset):

    def __init__(self, config, args):
        super(CLIP_CC3M_dataset, self).__init__()

        self.config = config

        self.img_files = []
        self.txt_files = []

        if args.retrain:
            self.img_files = [os.path.join(config.cc3m_dir + 'origin_img', f) for f in os.listdir(config.cc3m_dir + 'origin_img')]
            self.txt_files = [os.path.join(config.cc3m_dir + 'origin_txt', f) for f in os.listdir(config.cc3m_dir + 'origin_txt')]
        else:
            self.img_files = [os.path.join(config.cc3m_dir + 'origin_img', f) for f in
                              os.listdir(config.cc3m_dir + 'origin_img')]
            self.img_files += [os.path.join(config.cc3m_dir + 'except_img', f) for f in
                               os.listdir(config.cc3m_dir + 'except_img')]
            self.txt_files = [os.path.join(config.cc3m_dir + 'origin_txt', f) for f in
                              os.listdir(config.cc3m_dir + 'origin_txt')]
            self.txt_files += [os.path.join(config.cc3m_dir + 'except_txt', f) for f in
                               os.listdir(config.cc3m_dir + 'except_txt')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        txt_path = self.txt_files[idx]

        return torch.load(img_path), torch.load(txt_path)