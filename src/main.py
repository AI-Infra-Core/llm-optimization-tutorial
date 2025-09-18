# main.py

import click, json
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import VirtualTokenDataset, packed_sequence_collate_fn
from trainer import Trainer
from modeling_qwen3 import Qwen3ForCausalLM


@click.command()
# Model parameters
@click.option('--config-path', type=str, default='configs/qwen3-0.6B.json', help='Model config path.', show_default=True)
# Training parameters
@click.option('--start-lr', type=float, default=1e-4, help='Starting learning rate (max LR).', show_default=True)
@click.option('--end-lr', type=float, default=1e-5, help='Ending learning rate (min LR).', show_default=True)
@click.option('--batch-size', type=int, default=4, help='Batch size per GPU.', show_default=True)
@click.option('--log-interval', type=int, default=4, help='Interval for logging training status.', show_default=True)
# Dataset parameters
@click.option('--num-samples', type=int, default=66, help='Number of samples in the dataset.', show_default=True)
@click.option('--sequence-length', type=int, default=4096, help='Sequence length of the model.', show_default=True)
def train(start_lr, end_lr, batch_size, log_interval, num_samples, sequence_length, config_path):
    """
    A simple PyTorch trainer for LLM optimization.
    This script runs the training process using parameters from command-line options.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    config = SimpleNamespace(**config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Using device: {device} ---")

    dataset = VirtualTokenDataset(num_samples, sequence_length, config.vocab_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=packed_sequence_collate_fn)
    
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

    total_steps = len(train_loader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=end_lr
    )
    print(f"--- Cosine LR Scheduler enabled: T_max={total_steps}, Max_LR={start_lr}, Min_LR={end_lr} ---")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        log_interval=log_interval
    )
    trainer.train_single_epoch(train_loader)
    
    print("\n--- Training Finished ---")

if __name__ == "__main__":
    train()