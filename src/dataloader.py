import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image

def read_annotation_file(folder):
    return open(f"{folder}/det/det.txt", "r").read().split("\n")[:-1]

def parse_annotations(annotations):
    return np.reshape([np.array(annotation.split(","), np.float32)[:7] for annotation in annotations], (-1, 7))

class MOT15Dataset(Dataset):
    def __init__(self, data_root, transforms):
        folders = glob(f"{data_root}/*")
        self.annotations = [parse_annotations(read_annotation_file(folder)) for folder in folders]
        self.image_paths = [glob(f"{folder}/img1/*.jpg") for folder in folders]
        self.transforms = transforms

    def get_index_set(self, index):
        set_lengths = [len(item) for item in self.image_paths]
        cumulative_lengths = np.concatenate([np.zeros(1), np.cumsum(set_lengths)])
        idx = np.where(index >= cumulative_lengths)[0][-1]
        set_index = np.int32(index - cumulative_lengths[idx])

        return self.image_paths[idx][set_index], self.annotations[idx][np.where(self.annotations[idx][:, 0] == (set_index + 1))]

    def get_bbox(self, annotation, x_scale, y_scale):
        # Annotation fields:
        # Frame, ID, topleft, width, height, confidence
        return (annotation[2] * x_scale,
                annotation[3] * y_scale,
                (annotation[2] + annotation[4]) * x_scale,
                (annotation[3] + annotation[5]) * y_scale,)

    def __len__(self):
        return np.sum([len(item) for item in self.image_paths])

    def __getitem__(self, idx):
        img_path, annotation = self.get_index_set(idx)
        image = decode_image(img_path)
        h_, w_, c_ = np.shape(image)
        image = self.transforms(image)
        h, w, c = image.shape

        targets = {"boxes": torch.tensor([self.get_bbox(item, h / h_, w / w_) for item in annotation]).view(-1, 4),
                   "labels" : torch.tensor([1 for _ in range(len(annotation))]).to(torch.int64)}

        return image, targets

def collate(samples):
    images = [sample[0] for sample in samples]
    targets = [sample[1] for sample in samples]

    return images, targets

def create_dataloader(folder, transforms, **kwargs):
    dataset = MOT15Dataset(folder, transforms)

    return DataLoader(dataset, collate_fn=collate, **kwargs)

