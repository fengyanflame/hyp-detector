from hyp_model import get_model
from FENETWORK import FENETWORK
import torch
model = get_model() # load model structure
model.load_state_dict(torch.load(r"model_epoch_1.pth")) # load pth
output1 = FENETWORK(model,context='茶红素是一类异质的酸性酚性色素的总称,在红茶和普洱茶（熟茶）中含量极为丰富。')
output2 = FENETWORK(model,context='凡普信是凡普金科旗下的一站式金融信息服务品牌,致力于为全国有消费、融资需求的个人用户提供定制化的金融信息服务,并通过商业保理和融资租赁等机构开展消费分期及融资租赁等服务。')
output3 = FENETWORK(model,context='广告样品费是指为扩大商品购销业务所支付的广告样品费用,如向社会宣传商品而没置的宣传栏、橱窗、板报、印刷宣传资料,在报刊、电台、电视台刊登、广播业务广告等支付的费用。')
print(output1)
print(output2)
print(output3)