from torch import nn
from transformers import BertModel
path = 'hfl/chinese-macbert-base'
model = BertModel.from_pretrained(path, add_pooling_layer=False)
class addition_layer(nn.Module):
    def __init__(self):
        super(addition_layer, self).__init__()
        self.model = model
        self.qa_outputs = nn.Linear(768, 4)

    def forward(self, x):
        x = self.model(**x)
        x = self.qa_outputs(x['last_hidden_state'])
        return x
def get_model():
    return addition_layer()
