from numpy.core.numeric import outer
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import BertForSequenceClassification

from pytorch_transformers import AutoModelForSequenceClassification

import torch
from torch import nn
from torch.distributions import normal
from pytorch_transformers import modeling_bert

# bert_model = BertForSequenceClassification(BertConfig())

model_checkpoint = "distilbert-base-uncased"

auto_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)


class VLBertEmbeddings(modeling_bert.BertEmbeddings):
    def __init__(self, config):
        super(VLBertEmbeddings, self).__init__(config)

        self.region_embed = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob))

        self.adv_pertubation_embed = nn.Sequential(
            nn.Linear(1024, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.hidden_dropout_prob))

    def forward(self, region_features, masked_captions, position_ids):

        region_features = self.region_embed(region_features)

        position_embeddings = self.position_embeddings(position_ids)

        words_embeddings = self.word_embeddings(masked_captions) + position_embeddings

        embeddings = torch.cat((region_features, words_embeddings), dim=1)

        return self.dropout(self.LayerNorm(embeddings))


class Discriminator(modeling_bert.BertPreTrainedModel):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)

        self.embedding_layer = VLBertEmbeddings(config)
        self.transformer = auto_model.distilbert.transformer
        self.pre_classifier = auto_model.pre_classifier
        self.classifier = auto_model.classifier
        #--------------------------------------------
        # self.encoder = bert_model.bert.encoder
        # self.pooler = bert_model.bert.pooler
        # self.dropout = bert_model.dropout
        # self.classifier = bert_model.classifier
        #--------------------------------------------


        self.head_mask = [None] * config.num_hidden_layers

        self.apply(self._init_weights)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def forward(self, captions, position_ids, region_features,
                attention_mask):
              
        embeddings = self.embedding_layer(
            region_features, captions, position_ids)
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        output = self.transformer(embeddings, head_mask=self.head_mask)

        print(output.shape);exit(0)

        #--------------------------------------------
        # hidden_states = self.encoder(embeddings, attention_mask, head_mask=self.head_mask)[0]
        # output = self.pooler(hidden_states)
        # output = self.dropout(output)
        # output = self.classifier(output)
        #--------------------------------------------


        return output

