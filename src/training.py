from tqdm import tqdm
import torch
from torch.optim import Adam
from torchvision.ops import nms, box_iou
import numpy as np


def evaluate(model, data_test, device):
    model.eval()
    cumulative_precision = 0.

    with torch.no_grad():
        for data in tqdm(data_test):
            sequences, targets_list = data
            sequences = sequences.to(device)
            targets_list = [[{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in target.items()} for target in targets] 
                           for targets in targets_list]

            outputs = model(sequences)

            precision = 0
            frame_count = 0
            for batch_idx, targets in enumerate(targets_list):
                if batch_idx < len(outputs):
                    output = outputs[batch_idx]
                    # Use the last frame target for evaluation
                    ground_truth = targets[-1]
                    
                    if len(output["boxes"]) == 0 or len(ground_truth["boxes"]) == 0:
                        continue
                    indices = nms(output["boxes"], output["scores"], 0.5)
                    if len(indices) > 0:
                        boxes = output["boxes"][indices, :]
                        iou, _ = torch.max(box_iou(boxes, ground_truth["boxes"]), dim=1)
                        precision += torch.mean(iou).detach().cpu().numpy()
                        frame_count += 1
        
            cumulative_precision += precision / max(frame_count, 1)
    return cumulative_precision / max(1, len(data_test))


def train_one_epoch(model, optimizer, data_train, device):
    model.train()
    running_loss = 0.
    last_loss = 0.
    progress_bar = tqdm(enumerate(data_train), total=len(data_train))

    for step, data in progress_bar:
        sequences, targets_list = data
        sequences = sequences.to(device)
        targets_list = [[{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in target.items()} for target in targets] 
                       for targets in targets_list]
        
        optimizer.zero_grad()

        losses = model(sequences, targets_list)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        last_loss = loss.detach().cpu().numpy().item()
        running_loss += last_loss
        progress_bar.set_postfix(loss=running_loss / (step + 1))
    return last_loss


def fit(model, data_train, data_test, device, n_epochs=100, learning_rate=1e-4):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        loss_train = train_one_epoch(model, optimizer, data_train, device)
        loss_eval = evaluate(model, data_test, device)
        print(f"Epoch: {epoch}, training loss: {loss_train}, mean precision: {loss_eval}")
    return model