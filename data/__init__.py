import os

from .dataset import COCOCaptionGeneratorDataset, collate_fn_train, collate_fn_test
from .discriminator_dataset import COCOCaptionDiscriminatorDataset, collate_fn_discriminator_train
from .tokenizer import EOS, MASK, PAD, num_tokens, tokenizer

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise