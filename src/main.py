# main.py

import click, json
import addict
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import VirtualTokenDataset, packed_sequence_collate_fn
from trainer import Trainer
from modeling_qwen3 import Qwen3ForCausalLM
import yaml
from tools.logging import init_logger, logger

@click.command()
@click.argument("training_config_file")
def train(training_config_file):
    """
    A simple PyTorch trainer for LLM optimization.
    This script runs the training process using parameters from command-line options.
    """
    with open(training_config_file, 'r', encoding='utf-8') as f:
        training_config = yaml.safe_load(f)
    training_config = addict.Dict(training_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"--- Using device: {device} ---")

    # model configuration
    with open(training_config.model.config_path, 'r') as f:
        model_config = json.load(f)
    model_config = addict.Dict(model_config)
    model = Qwen3ForCausalLM(model_config).to(torch.bfloat16).to(device)

    # dataset configuration
    dataset_config = training_config.dataset
    dataset = VirtualTokenDataset(dataset_config.num_samples, dataset_config.sequence_length, model_config.vocab_size)
    train_loader = DataLoader(dataset, batch_size=dataset_config.batch_size, shuffle=True, collate_fn=packed_sequence_collate_fn)
    
    # optimizer configuration
    optimizer_config = training_config.optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.start_lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=optimizer_config.total_steps,
        eta_min=optimizer_config.end_lr
    )
    logger.info(f"--- Cosine LR Scheduler enabled: T_max={optimizer_config.total_steps}, Max_LR={optimizer_config.start_lr}, Min_LR={optimizer_config.end_lr} ---")

    # start training
    trainer = Trainer(
        training_config=training_config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        log_interval=training_config.logging.log_interval
    )
    trainer.train_single_epoch(train_loader)
    
    logger.info("\n--- Training Finished ---")

if __name__ == "__main__":
    init_logger()
    train()