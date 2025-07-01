Temporal Model Integration
Implemented ConvGRU and Plain GRU temporal processors
Both maintain state across sequences and support state reset

Enhanced Dataloader
Always returns sequences (batch, seq_len, C, H, W)
Maintains frame order while shuffling sequence batches
Supports both training and inference modes

Two Training Scripts
run_conv_gru.py - For ConvGRU model training
run_simple_gru.py - For Plain GRU model training
