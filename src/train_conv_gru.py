import torch
from torchvision.transforms import v2

from model import construct_temporal_model_conv_gru
from dataloader import create_dataloader
from training import fit

def main():
    """Run training with ConvGRU temporal processing"""
    
    transforms = v2.Compose([
        v2.Resize(size=(480, 320)),  # Reduced resolution for memory efficiency
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Using ConvGRU temporal processor")

    root = "path/to/MOT15"  
    data_train = create_dataloader(f"{root}/train", transforms=transforms, batch_size=1, shuffle=True, sequence_length=4)
    data_test = create_dataloader(f"{root}/test", transforms=transforms, batch_size=1, shuffle=False, sequence_length=4)

    # Create temporal model with ConvGRU processing
    model = construct_temporal_model_conv_gru(num_classes=5, hidden_channels=64)

    model = fit(model, data_train, data_test, device, n_epochs=2, learning_rate=1e-4)
    
    print("ConvGRU training completed successfully!")
    return model

if __name__ == "__main__":
    main()
