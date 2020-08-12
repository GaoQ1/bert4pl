from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class BertForLeam(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.label_embedding = nn.Embedding(config.num_labels, config.hidden_size)
        self.ul_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.ul_activation = nn.ReLU()

        self.init_weights()

    @staticmethod
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        sequence_output, pooled_output = outputs[:2]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            label_embedding = self.label_embedding(labels)
            gather = torch.bmm(label_embedding.unsqueeze(2), pooled_output.unsqueeze(1))

            label_embedding_l2 = torch.norm(label_embedding, p=2, dim=0) #l2正则
            pooled_output_l2 = torch.norm(pooled_output, p=2, dim=0) #l2正则

            g_triangle = torch.matmul(label_embedding_l2.unsqueeze(1), pooled_output_l2.unsqueeze(0))
            gather = gather / g_triangle.unsqueeze(0)

            ul = self.ul_activation(self.ul_linear(gather))
            ml = self.kmax_pooling(ul, 1, 1)
            beta = nn.Softmax(dim=-1)(ml).squeeze()

            pooled_output = self.dropout(beta * pooled_output)
            logits = self.classifier(pooled_output)

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits, ) + outputs[2:]
        else:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            outputs = (logits, ) + outputs[2:]


        return outputs  # (loss), logits, (hidden_states), (attentions)
    