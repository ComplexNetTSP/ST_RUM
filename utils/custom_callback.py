import time
from pytorch_lightning import Callback, LightningModule, Trainer


class TimingCallback(Callback):
    def __init__(self):
        self.epoch_start = None
        self.batch_start = None
        self.train_batch_times = []
        self.test_batch_times = []
        self.predict_batch_times = []
        self.all_epoch_times = []
        self.all_train_batch_times = []
        
    # Training
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.epoch_start = time.time()
        self.train_batch_times = []
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx):
        self.batch_start = time.time()
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        if self.batch_start:
            self.train_batch_times.append(time.time() - self.batch_start)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.epoch_start and self.train_batch_times:
            epoch_time = time.time() - self.epoch_start
            avg_batch_time = sum(self.train_batch_times) / len(self.train_batch_times)
            
            # Store for overall statistics
            self.all_epoch_times.append(epoch_time)
            self.all_train_batch_times.extend(self.train_batch_times)
            
            # epoch_per_sec = 1.0 / epoch_time
            # batch_per_sec = 1.0 / avg_batch_time
            
            # print(f"Train Epoch {trainer.current_epoch}: {epoch_per_sec:.4f} epoch/s, {batch_per_sec:.2f} batch/s")
    
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.all_epoch_times and self.all_train_batch_times:
            total_epochs = len(self.all_epoch_times)
            total_time = sum(self.all_epoch_times)
            avg_epoch_time = total_time / total_epochs
            avg_batch_time = sum(self.all_train_batch_times) / len(self.all_train_batch_times)
            
            avg_epoch_per_sec = 1.0 / avg_epoch_time
            avg_batch_per_sec = 1.0 / avg_batch_time
            
            print(f"\nTraining Summary: {total_epochs} epochs, {total_time:.2f}s total")
            print(f"Average: {avg_epoch_per_sec:.4f} epoch/s, {avg_batch_per_sec:.2f} batch/s")
    
    # Test
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        self.test_start = time.time()
        self.test_batch_times = []
    
    def on_test_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx=0):
        self.batch_start = time.time()
    
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        if self.batch_start:
            self.test_batch_times.append(time.time() - self.batch_start)
    
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.test_batch_times:
            total_time = time.time() - self.test_start
            avg_batch_time = sum(self.test_batch_times) / len(self.test_batch_times)
            batch_per_sec = 1.0 / avg_batch_time
            print(f"Test: {batch_per_sec:.2f} batch/s, Total: {total_time:.2f}s")
    
    # Predict
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule):
        self.predict_start = time.time()
        self.predict_batch_times = []
    
    def on_predict_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, dataloader_idx=0):
        self.batch_start = time.time()
    
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        if self.batch_start:
            self.predict_batch_times.append(time.time() - self.batch_start)
    
    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.predict_batch_times:
            total_time = time.time() - self.predict_start
            avg_batch_time = sum(self.predict_batch_times) / len(self.predict_batch_times)
            batch_per_sec = 1.0 / avg_batch_time
            print(f"Predict: {batch_per_sec:.2f} batch/s, Total: {total_time:.2f}s")