from hyp_model import get_model
from FENETWORK import FENETWORK
import torch
model = get_model() # load model structure
model.load_state_dict(torch.load(r"./model_epoch_1.pth")) # load pth
output1 = FENETWORK(model,context='茶红素是一类异质的酸性酚性色素的总称,在红茶和普洱茶（熟茶）中含量极为丰富。')
output2 = FENETWORK(model,context='证券时报记者 吴少龙8月13日，由中国REITs 50人论坛携手证券时报、新财富等单位举办的“首届大湾区基础设施REITs高峰论坛”在深圳顺利举行')
output3 = FENETWORK(model,context='2017年县域门诊量453.7万人次，同比增长9.16%住院5.6万人次，同比增长7.89%2017年基层医疗机构门急诊217.1万人次，同比增长9.83%，基层门急诊人次占比提升至47.8%')
print(output1)
print(output2)
print(output3)