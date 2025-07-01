import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

def read_annotation_file(folder):
    try:
        return open(f"{folder}/det/det.txt", "r").read().split("\n")[:-1]
    except:
        return []

def parse_annotations(annotations):
    if not annotations:
        return np.array([]).reshape(0, 7)
    return np.reshape([np.array(annotation.split(","), np.float32)[:7] for annotation in annotations], (-1, 7))

class MOT15Dataset(Dataset):
    def __init__(self, data_root, transforms, sequence_length=10):
        folders = glob(f"{data_root}/*")
        self.annotations = [parse_annotations(read_annotation_file(folder)) for folder in folders]
        self.image_paths = [sorted(glob(f"{folder}/img1/*.jpg")) for folder in folders]
        self.transforms = transforms
        self.sequence_length = sequence_length
        
        # Create sequence indices
        self.sequences = []
        for folder_idx, paths in enumerate(self.image_paths):
            for i in range(len(paths) - sequence_length + 1):
                self.sequences.append((folder_idx, i))

    def get_bbox(self, annotation, x_scale, y_scale):
        # Annotation fields: Frame, ID, topleft, width, height, confidence
        return (annotation[2] * x_scale,
                annotation[3] * y_scale,
                (annotation[2] + annotation[4]) * x_scale,
                (annotation[3] + annotation[5]) * y_scale,)

    def get_frame_data(self, folder_idx, frame_idx):
        img_path = self.image_paths[folder_idx][frame_idx]
        frame_num = frame_idx + 1
        
        # Get annotations for this frame
        annotations = self.annotations[folder_idx]
        frame_annotations = annotations[annotations[:, 0] == frame_num] if len(annotations) > 0 else []
        
        image = decode_image(img_path)
        h_, w_ = image.shape[1], image.shape[2]
        image = self.transforms(image)
        h, w = image.shape[1], image.shape[2]
        
        # Create targets
        if len(frame_annotations) > 0:
            boxes = torch.tensor([self.get_bbox(item, w / w_, h / h_) for item in frame_annotations]).view(-1, 4)
            labels = torch.tensor([1 for _ in range(len(frame_annotations))]).to(torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        targets = {"boxes": boxes, "labels": labels}
        return image, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        folder_idx, start_frame = self.sequences[idx]
        
        images = []
        targets = []
        
        for i in range(self.sequence_length):
            image, target = self.get_frame_data(folder_idx, start_frame + i)
            images.append(image)
            targets.append(target)
        
        return torch.stack(images), targets

def collate(samples):
    image_sequences = []
    target_sequences = []
    
    for img_seq, tgt_seq in samples:
        image_sequences.append(img_seq)
        target_sequences.append(tgt_seq)
    
    return torch.stack(image_sequences), target_sequences

def create_dataloader(folder, transforms, sequence_length=10, **kwargs):
    dataset = MOT15Dataset(folder, transforms, sequence_length)
    return DataLoader(dataset, collate_fn=collate, **kwargs)

