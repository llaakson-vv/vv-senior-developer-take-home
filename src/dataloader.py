import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
import numpy as np
from glob import glob

class MOTSequenceDataset(Dataset):
    def __init__(self, data_root, transforms, sequence_length=10):
        self.transforms = transforms
        self.sequence_length = sequence_length
        
        folders = glob(f"{data_root}/*")
        self.sequences = []
        
        for folder in folders:
            image_paths = sorted(glob(f"{folder}/img1/*.jpg"))
            annotations = self._read_annotations(folder)
            
            for i in range(len(image_paths) - sequence_length + 1):
                seq_data = {
                    'images': image_paths[i:i+sequence_length],
                    'annotations': annotations,
                    'start_frame': i + 1
                }
                self.sequences.append(seq_data)
    
    def _read_annotations(self, folder):
        try:
            with open(f"{folder}/det/det.txt", "r") as f:
                lines = f.read().split("\n")[:-1]
            return np.reshape([np.array(line.split(","), np.float32)[:7] for line in lines], (-1, 7))
        except:
            return np.array([]).reshape(0, 7)
    
    def _get_frame_targets(self, annotations, frame_num, h_scale, w_scale):
        frame_annotations = annotations[annotations[:, 0] == frame_num]
        
        if len(frame_annotations) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        
        boxes = []
        for ann in frame_annotations:
            x1, y1, w, h = ann[2:6]
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale])
        
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64)
        }
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        images = []
        targets = []
        
        for i, img_path in enumerate(seq_data['images']):
            image = decode_image(img_path)
            h_orig, w_orig = image.shape[1], image.shape[2]
            
            image = self.transforms(image)
            h_new, w_new = image.shape[1], image.shape[2]
            
            frame_num = seq_data['start_frame'] + i
            frame_targets = self._get_frame_targets(
                seq_data['annotations'], 
                frame_num,
                w_new / w_orig,
                h_new / h_orig
            )
            
            images.append(image)
            targets.append(frame_targets)
        
        return torch.stack(images), targets

def create_dataloader(folder, transforms, sequence_length=10, batch_size=2, shuffle=True):
    dataset = MOTSequenceDataset(folder, transforms, sequence_length)
    
    def collate_fn(batch):
        image_sequences = []
        target_sequences = []
        
        for img_seq, tgt_seq in batch:
            image_sequences.append(img_seq)
            target_sequences.append(tgt_seq)
        
        return torch.stack(image_sequences), target_sequences
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0
    )

