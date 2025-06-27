# Verso senior ML-developer take-home assignment

The purpose of this assignment is to evaluate the ability of the candidate to build and customize model architectures and training infrastructure in PyTorch framework.
The assignment is three-fold: 
- modify the dataloader to return sequences of 10 samples instead of individual samples. I.e., the input shape should be (batches, 10, channels, height, width).
- add a trainable stateful layers (e.g., RNN) to the provided RCNN-model to leverage the sequential nature of the dataset in detection.
- modify the training script to support the sequential data and the stateful model.

The assignment is designed to take 2-3 hours to complete. The preferred method of submission is publishing the solution on a personal github page and providing a link.

IMPORTANT NOTES:
- The model is a detection/classification model and thus returns the class labels. The classification outputs and targets are not relevant for this task and can be ignored.
- The training isn't expected to converge; it is sufficient to show that the custom model is training (i.e., the gradients are backpropagated and the optimizer updates the layer weights).
- If GPU memory becomes an issue, the sequence lengths or images and the model inputs can be resized as necessary.

## Pre-requisites

- Install the libraries in the project requirements
- Download the following dataset
```bash
https://motchallenge.net/data/MOT15/
```

## The dataset

`MOT15` is an object tracking dataset consisting of several labelled image sequences of moving people.

## The dataloader

A dataloader for reading and organizing the dataset is provided in the repository. The first task is to modify the dataloader such that its iterator returns a sequence instead of singular samples.
The training dataloader should shuffle the samples. However, the ordering of the frames inside a sample sequence should remain intact. Similarly, the dataloader should be able to provide batched data.

## The model

The relevant parts of the `torchvision` R-CNN- model implementation is provided in the repository for convenience.
Add a new module to the model that has the following features:
- It is trained as a part of the model backpropagation.
- It has a state or otherwise stores information on previously seen inputs.
- It influences the model outputs.
- The state can be reset and restored; i.e., previous sequence's state doesn't influence a new sequence, and several sequences could be processed in a round-robin fashion.

