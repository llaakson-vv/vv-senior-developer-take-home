from tqdm import tqdm
import torch
from torch.optim import Adam
from torchvision.ops import nms, box_iou
import numpy as np


def evaluate(model, data_test):
    model.eval()
    cumulative_precision = 0.

    with torch.no_grad():
        for data in tqdm(data_test):
            inputs, targets = data

            _,  outputs = model(inputs)

            precision = 0
            for output, ground_truth in zip(outputs, targets):
                indices = nms(output["boxes"], output["scores"], 0.5)

                if len(indices) > 0:
                    boxes = output["boxes"][indices, :]
                    iou, _ = torch.max(box_iou(boxes, ground_truth["boxes"]), dim=1)
                    precision += torch.mean(iou).detach().numpy()
        
            cumulative_precision += precision / max(len(outputs), 1)
    return cumulative_precision / max(1, len(data_test))


def train_one_epoch(model, optimizer, data_train):
    model.train()
    running_loss = 0.
    last_loss = 0.
    progress_bar = tqdm(enumerate(data_train), total=len(data_train))

    optimizer = Adam(model.parameters(), lr=1e-4)

    for step, data in progress_bar:
        inputs, targets = data
        optimizer.zero_grad()

        losses, detections = model(inputs, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        last_loss = loss.detach().to(device="cpu").numpy().item()
        running_loss += last_loss
        progress_bar.set_postfix(loss=running_loss / (step + 1))
        progress_bar.update()
    return last_loss


def fit(model, data_train, data_test, n_epocs=100, learning_rate=1e-4):
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epocs):
        loss_train = train_one_epoch(model, optimizer, data_train)
        loss_eval = evaluate(model, data_test)

        print(f"Epoch: {epoch}, training loss: {loss_train}, mean precision: {loss_eval}")
    return model


