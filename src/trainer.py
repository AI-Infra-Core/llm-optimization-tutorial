# trainer.py

import torch
import time

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, log_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.log_interval = log_interval

    def train_single_epoch(self, train_loader):
        """
        Runs a single training epoch.
        """
        print("\n--- Starting Single Epoch Training ---")
        self.model.train()
        
        running_loss = 0.0
        total_steps = len(train_loader)

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

            time_end = time.time()
            
            running_loss += loss.item()

            if (step + 1) % self.log_interval == 0:
                print(f"  Step [{step + 1}/{total_steps}], Loss: {loss.item():.4f}, Step Time: {time_end - time_start:.3f}s")
        
        epoch_loss = running_loss / total_steps
        print(f"--- Single Epoch Average Loss: {epoch_loss:.4f} ---")
