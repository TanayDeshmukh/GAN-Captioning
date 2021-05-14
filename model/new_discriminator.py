from numpy.core.numeric import outer
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers import BertForSequenceClassification
from pytorch_transformers import AutoModelForSequenceClassification


import torch
from torch import nn
from torch.distributions import normal
from pytorch_transformers import modeling_bert


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


class NewDiscriminator(modeling_bert.BertPreTrainedModel):
    def __init__(self, config):
        super(NewDiscriminator, self).__init__(config)

        self.embedding_layer = VLBertEmbeddings(config)
        self.encoder = modeling_bert.BertEncoder(config)
        self.classifier = modeling_bert.BertLMPredictionHead(config)
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
        
        # batch_size = region_features.shape[0]
      
        embeddings = self.embedding_layer(
            region_features, captions, position_ids)

        # print(self.classifier);exit(0)
        
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        model_checkpoint = "distilbert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        
        print(model)
        exit(0)

        model = BertForSequenceClassification(BertConfig())
        encoder = model.bert.encoder
        pooler = model.bert.pooler
        dropout = model.dropout
        classifier = model.classifier


        output = encoder(embeddings, attention_mask, head_mask=self.head_mask)
        # print(type(output))
        # print(len(output));exit(0)
        output = pooler(output[0])
        output = dropout(output)
        output = classifier(output)

        print(output.shape)
        print(output)
        exit(0)

        hidden_states = self.encoder(embeddings, attention_mask, self.head_mask)[0]

        output = self.classifier(hidden_states)

        print(hidden_states.shape, output.shape)
        exit(0)

        return self.classifier(hidden_states)

