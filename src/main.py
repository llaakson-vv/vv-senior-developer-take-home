import torch
from torchvision.transforms import v2

from model import construct_model 
from dataloader import create_dataloader
from training import fit

transforms = v2.Compose([
    v2.Resize(size=(640, 480)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

root = "/<path>/<to>/MOT15"
data_train = create_dataloader(f"{root}/train", transforms=transforms, batch_size=8, shuffle=True)
data_test = create_dataloader(f"{root}/test", transforms=transforms, batch_size=8, shuffle=False)
model = construct_model()

model = fit(model, data_train, data_test, 100)
