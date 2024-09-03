import satlaspretrain_models
import torch

weights_manager = satlaspretrain_models.Weights()

model = weights_manager.get_pretrained_model("Landsat_SwinB_SI", head=satlaspretrain_models.Head.SEGMENT, fpn=True, num_categories=1)
tensor = torch.zeros((1, 11, 512, 512), dtype=torch.float32)

model.eval()
output = model(tensor)
print(output)