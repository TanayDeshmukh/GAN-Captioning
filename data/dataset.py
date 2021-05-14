import json
import h5py
import torch
import random
import numpy as np
import os.path as osp

from config import _C as config
from collections import namedtuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import EOS, MASK, PAD, tokenizer

TrainSample = namedtuple("TrainSample", ["caption", "all_captions", "cocoid", "file_name"])
InfrSample = namedtuple("InfrSample", ["all_captions", "cocoid", "file_name"])
# TestSample = namedtuple("Sample", ["caption", "cocoid", "file_name"])

class COCOCaptionGeneratorDataset(Dataset):

    def __init__(self, root, split, max_detections=50, sort_by_prob=False):
        self.split = split
        self.root = root
        self.max_detections = max_detections
        self.sort_by_prob = sort_by_prob

        if self.split == 'test':
            self.load_fn = self._get_item_infer
            self.build_infer_samples()
        else:
            self.load_fn = self._get_item_train
            self.build_train_samples()

    def build_train_samples(self):
        with open(osp.join(self.root, 'dataset_coco.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_train_data = osp.join('./data/generated_files', f'{self.split}_data.pth')
        if not osp.exists(file_train_data):
            print("Generating samples...")
            samples = list()
            for item in captions:
                if item['split'] in self.split:
                    file_name = item['filename']
                    cocoid = item['cocoid']
                    all_captions = []
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        all_captions.append(caption)
                        caption = tokenizer.encode(caption)
                        sample = TrainSample(caption=caption, all_captions=all_captions, cocoid=cocoid, file_name=file_name)
                        samples.append(sample)
            torch.save(samples, file_train_data)
        else:
            print("Loading from generated samples...")
            samples = torch.load(file_train_data)

        self.samples = samples
    
    def build_infer_samples(self):
        with open(osp.join(self.root, 'dataset_coco.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_train_data = osp.join('./data/generated_files', f'{self.split}_data.pth')
        if not osp.exists(file_train_data):
            samples = list()
            for item in captions:
                if item['split'] in self.split:
                    file_name = item['filename']
                    cocoid = item['cocoid']
                    all_captions = []
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        all_captions.append(caption)
                    
                    sample = InfrSample(all_captions=all_captions, cocoid=cocoid, file_name=file_name)
                    samples.append(sample)

            torch.save(samples, file_train_data)
        else:
            samples = torch.load(file_train_data)

        self.samples = samples
    
    def get_image_features(self, cocoid):
        with h5py.File(config.detections_file, 'r') as f:
            precomp_features = f['%d_features' % cocoid][()]

            if self.sort_by_prob:
                precomp_features = precomp_features[np.argsort(np.max(f['%d_cls_prob' % cocoid][()], -1))[::-1]]
            
        delta = self.max_detections - precomp_features.shape[0]
        if delta > 0:
            precomp_features = np.concatenate([precomp_features, np.zeros((delta, precomp_features.shape[1]))], axis=0)
        elif delta < 0:
            precomp_features = precomp_features[:self.max_detections]
         
        precomp_features = precomp_features.astype(np.float32)

        return torch.from_numpy(precomp_features)

    def __getitem__(self, index):
        return self.load_fn(index)
    
    def _get_item_train(self, index):
        sample = self.samples[index]

        caption = sample.caption
        image_name = sample.file_name
        cocoid = sample.cocoid
        all_captions = sample.all_captions

        caption = caption + [EOS]

        masked_caption = caption.copy()
        num_masks = random.randint(1, len(caption))
        selected_idx = random.sample(range(len(caption)), num_masks)
        for i in selected_idx:
            masked_caption[i] = MASK
        
        caption = torch.tensor(caption, dtype=torch.long)
        masked_caption = torch.tensor(masked_caption, dtype=torch.long)

        image_features = self.get_image_features(cocoid)

        return caption, masked_caption, image_features, image_name, all_captions


    def _get_item_infer(self, index):
        sample = self.samples[index]

        cocoid = sample.cocoid
        all_captions = sample.all_captions
        file_name = sample.file_name

        image_features = self.get_image_features(cocoid)

        return image_features, all_captions, file_name

    def __len__(self):
        return len(self.samples)

def collate_fn_train(batch):

    batch = list(zip(*batch))

    captions = pad_sequence(batch[0], batch_first=True, padding_value=PAD)
    masked_captions = pad_sequence(batch[1], batch_first=True, padding_value=PAD)
    
    image_features = torch.stack(batch[2], dim=0)
    image_names = batch[3]
    all_captions = batch[4]

    return captions, masked_captions, image_features, image_names, all_captions

def collate_fn_test(batch):

    batch = list(zip(*batch))

    image_features = torch.stack(batch[0], dim=0)
    all_captions = batch[1]
    file_name = batch[2]

    return image_features, all_captions, file_name
