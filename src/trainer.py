# trainer.py

import torch
import time
import numpy as np
from tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from tools.logging import logger
from tools.utils import MFUCalculator
class Trainer:
    def __init__(self, training_config, model, optimizer, criterion, scheduler, device, log_interval=10):
        self.training_config = training_config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval
        self.mfu_calculator = MFUCalculator(model)

    def train_single_epoch(self, train_loader):
        """
        Runs a single training epoch.
        """
        logger.info("\n--- Starting Single Epoch Training ---")
        self.model.train()
        
        running_loss = 0.0
        total_steps = len(train_loader)

        with (
            maybe_enable_profiling(self.training_config.profiling) as torch_profiler,
            maybe_enable_memory_snapshot(self.training_config.profiling) as memory_profiler,
        ):
            for step, batch in enumerate(train_loader):
                time_start = time.time()
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                cu_seqlens = batch['cu_seqlens'].to(self.device)
                position_ids = batch['position_ids'].to(self.device)
                max_seqlen = batch['max_seqlen']

                outputs = self.model(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids
                )

                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                torch.cuda.synchronize()

                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                time_this_step = time.time() - time_start

                # metrics logging
                seq_lengths = np.diff(cu_seqlens.cpu())
                total_seq_length = np.sum(seq_lengths)
                seq_length_squared_sum = np.sum(seq_lengths * seq_lengths)
                self_weighted_seq_length = seq_length_squared_sum / total_seq_length
                loss_this_step = loss.item()
                tps = total_seq_length / time_this_step
                mfu = self.mfu_calculator.calculate(total_seq_length, seq_length_squared_sum, time_this_step)
                
                running_loss += running_loss

                if (step + 1) % self.log_interval == 0:
                    logger.info(f"  Step [{step + 1}/{total_steps}], Loss: {loss_this_step:.4f}, Self_weighted_seq_length: {self_weighted_seq_length:.2f}, Step Time: {time_this_step:.3f}s, TPS: {tps:.2f}, MFU: {mfu:.2f}")
            
            epoch_loss = running_loss / total_steps
            logger.info(f"--- Single Epoch Average Loss: {epoch_loss:.4f} ---")
