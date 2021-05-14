import torch
import random
from os import EX_IOERR

# from model import Discriminator, DistilBertModel
from model import Discriminator
from config import _C as config
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from data import EOS, MASK, PAD, discriminator_dataset, tokenizer, num_tokens
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import AutoModelForSequenceClassification
from pytorch_transformers import BertForSequenceClassification, BertTokenizer
from pytorch_transformers.modeling_distilbert import DistilBertPreTrainedModel, Transformer, DistilBertConfig
from data import COCOCaptionDiscriminatorDataset, collate_fn_discriminator_train

train_dataset = COCOCaptionDiscriminatorDataset(
        root = config.root_dir,
        split = 'train'
    )

train_data_loader = DataLoader(
        train_dataset,
        batch_size = config.train_batch_size,
        shuffle = True,
        num_workers = 0,
        drop_last = True,
        collate_fn = collate_fn_discriminator_train
    )

model_checkpoint = "distilbert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
print(model.distilbert.transformer)
print(model.pre_classifier)
print(model.classifier)

exit(0)

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# print(model)
# exit(0)

# config = DistilBertConfig()
# model = DistilBertModel(config)
# print(model)
# print(config.num_hidden_layers)
# exit(0)

bert_config = BertConfig()
discriminator = Discriminator(bert_config)

# print(discriminator);exit(0)

device = "cpu"

for idx, (captions, image_features, image_name, label) in enumerate(train_data_loader):
    
    captions = captions.to(device)  # (N, L), long
    image_features = image_features.to(device)  # (N, L), long

    num_img_tokens = image_features.size(1)
    seq_length = captions.size(1)
    batch_size = captions.size(0)

    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(captions)

    attention_mask = (captions != PAD).float()
    _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
    attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

    pred_scores = discriminator(
            captions,
            position_ids,
            image_features,
            attention_mask)
    
    print(pred_scores)
    print(label)
        
    exit(0)