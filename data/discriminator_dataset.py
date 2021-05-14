import json
import h5py
import copy
import torch
import random
import numpy as np
import os.path as osp

from tqdm import tqdm
from config import _C as config
from collections import namedtuple
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import EOS, MASK, PAD, tokenizer

TrainSample = namedtuple("TrainSample", ["caption", "cocoid", "file_name", "label"])
InfrSample = namedtuple("InfrSample", ["caption", "cocoid", "file_name", "label"])
# TestSample = namedtuple("Sample", ["caption", "cocoid", "file_name"])

class COCOCaptionDiscriminatorDataset(Dataset):

    def __init__(self, root, split, max_detections=50, sort_by_prob=False):
        self.split = split
        self.root = root
        self.max_detections = max_detections
        self.sort_by_prob = sort_by_prob

        self.dset_captions = dset.CocoCaptions(root='/netscratch/karayil/mscoco/data/'+split+'2014',
                                annFile="/netscratch/karayil/mscoco/data/annotations/captions_"+split+"2014.json",
                                transform=transforms.ToTensor())
        
        self.coco = self.dset_captions.coco

        self.build_samples()

    def build_samples(self):
        
        print('Loading captions..')
        with open(osp.join(self.root, 'dataset_coco.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_train_data = osp.join('./data/generated_files', f'{self.split}_discriminator_data.pth')
        if not osp.exists(file_train_data):
            print("Generating samples...")
            samples = list()
            for item in tqdm(captions):
                if item['split'] in self.split:
                    file_name = item['filename']
                    cocoid = item['cocoid']
                    temp = copy.deepcopy(self.dset_captions.ids)
                    temp.remove(cocoid)
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        caption = tokenizer.encode(caption)

                        if len(temp) == 0:
                            print(cocoid)
                            continue
                        
                        other_id = random.choice(temp)
                        ann_ids = self.coco.getAnnIds(imgIds=other_id)
                        anns = self.coco.loadAnns(ann_ids)
                        target = [ann['caption'] for ann in anns]
                        other_caption = target[random.choice(range(5))]                        
                        other_caption = tokenizer.encode(other_caption)

                        correct_sample = TrainSample(caption=caption,
                                                cocoid=cocoid, 
                                                file_name=file_name,
                                                label=1)                        
                        samples.append(correct_sample)

                        wrong_sample = TrainSample(caption=other_caption,
                                                cocoid=cocoid, 
                                                file_name=file_name,
                                                label=0)
                        samples.append(wrong_sample)

            torch.save(samples, file_train_data)
        else:
            print("Loading from generated samples...")
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
        sample = self.samples[index]

        caption = sample.caption
        image_name = sample.file_name
        cocoid = sample.cocoid
        label = sample.label

        caption = caption + [EOS]
        
        caption = torch.tensor(caption, dtype=torch.long)

        image_features = self.get_image_features(cocoid)

        return caption, image_features, image_name, label


    def __len__(self):
        return len(self.samples)

def collate_fn_discriminator_train(batch):

    batch = list(zip(*batch))

    # caption, image_features, image_name, label

    captions = pad_sequence(batch[0], batch_first=True, padding_value=PAD)
    # other_captions = pad_sequence(batch[1], batch_first=True, padding_value=PAD)
    
    image_features = torch.stack(batch[1], dim=0)
    image_names = batch[2]
    label = torch.tensor(batch[3])

    return captions, image_features, image_names, label
