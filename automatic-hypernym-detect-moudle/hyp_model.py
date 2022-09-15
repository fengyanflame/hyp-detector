from torch import nn
from transformers import BertModel


class Hyp_Model(nn.Module):
    def __init__(self):
        path = 'hfl/chinese-macbert-base'
        super(Hyp_Model, self).__init__()
        self.model = BertModel.from_pretrained(path, add_pooling_layer=False,hidden_dropout_prob=0.2,attention_probs_dropout_prob=0.2)
        self.qa_outputs = nn.Linear(768, 4)

    def forward(self, x):
        x = self.model(**x)
        x = self.qa_outputs(x['last_hidden_state'])
        return x


def get_model():
    return Hyp_Model()
