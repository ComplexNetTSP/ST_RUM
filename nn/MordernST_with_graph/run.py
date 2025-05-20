from tsl.datasets import MetrLA, AirQuality, PemsBay
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from einops import rearrange, repeat
from torch_geometric.utils.undirected import is_undirected
from tsl.engines import Imputer, Predictor
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule,TemporalSplitter
from tsl.data.preprocessing import StandardScaler
import torch
from tsl.metrics import torch as torch_metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tsl.data.preprocessing import StandardScaler, RobustScaler
from pytorch_lightning import Trainer
import random
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from modernST import ModernTCN_rum
import argparse
import os


parser = argparse.ArgumentParser(description='ModernST')

# random seed
parser.add_argument('--random_seed', type=int, default=42, help='random seed')

# data loader
parser.add_argument('--data', type=str, default='la', help='dataset')
parser.add_argument('--directed', type=str, help='directed graph or undirected graph')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold for gaussian kernel distance')

# forecasting task
parser.add_argument('--window', type=int, default=12, help='input training window')
parser.add_argument('--horizon', type=int, default=12, help='prediction horizon length')

#ModernST
parser.add_argument('--input_size', type=int, default=1, help='input size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--exog_size', type=int, default=1, help='exogenous size')
parser.add_argument('--large_kernel', nargs='+',type=int, default=[7,5,3], help='large kernel')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--rw_sample', type=int, default=20, help='random walk samples')
parser.add_argument('--rw_length', type=int, default=5, help='random walk length')


# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--limit_train_batches', type=int, default=150, help='limit train batches')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')


# GPU
parser.add_argument('--accelerator', type=str, default='gpu', help='gpu or cpu')
parser.add_argument('--devices', type=int, default=0, help='device ids of multile gpus')



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    return seed

def get_model_log_name(model, torch_dataset):
        class_name = model.__class__.__name__
        directed = str(not is_undirected(torch_dataset.edge_index))
        return f"{class_name}_directed_{directed}_with_graph"

def get_dateset(dataset_name):
    if dataset_name == 'la':
        dataset = MetrLA(root='./data/metrla')
    elif dataset_name == 'bay':
        dataset = PemsBay(root='./data/bay', mask_zeros=True)
    elif dataset_name == 'aq':
        dataset = AirQuality(root='./data/aq', impute_nans=True, small=False)
    elif dataset_name == 'aq36':
        dataset = AirQuality(root='./data/aq36', impute_nans=True, small=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    
    return dataset

def get_splitter(dataset_name):
    # Air quality datasets
    if dataset_name in ['aq', 'aq36']:
        return TemporalSplitter(val_len=0.1, test_len=0.2)
    
    # Traffic datasets
    elif dataset_name in ['la', 'bay']:
        return TemporalSplitter(val_len=0.1, test_len=0.2)


def experiment(arg):
    seed_everything(arg.random_seed)

    dataset = get_dateset(dataset_name=arg.data)

    connectivity = dataset.get_connectivity(threshold=arg.threshold,
                                            include_self=False,
                                            force_symmetric=not arg.directed,
                                            layout="edge_index")

    covariates = {'u': dataset.datetime_encoded('day').values}

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                        connectivity=connectivity,
                                        mask=dataset.mask,
                                        covariates=covariates,
                                        window=arg.window,
                                        horizon=arg.horizon)
    
    # Normalize data using mean and std computed over time and node dimensions
    scalers = {'target': StandardScaler(axis=(0, 1))}

    # Split data sequentially:
    #   |------------ dataset -----------|
    #   |--- train ---|- val -|-- test --|
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=get_splitter(arg.data),
        batch_size=arg.batch_size,
        workers=arg.num_workers
    )

    dm.setup()
    

    loss_fn = torch_metrics.MaskedMAE()
    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mae_step_2': torch_metrics.MaskedMAE(at=2),
        'mae_step_3': torch_metrics.MaskedMAE(at=5),
        'mae_step_4': torch_metrics.MaskedMAE(at=11),
        'mse_step_2': torch_metrics.MaskedMSE(at=2),
        'mse_step_3': torch_metrics.MaskedMSE(at=5),
        'mse_step_4': torch_metrics.MaskedMSE(at=11)
    }



    model = ModernTCN_rum(input_size=arg.input_size,
                          hidden_size = arg.hidden_size,
                          patch_size = (1, arg.rw_sample, arg.rw_length+1),
                          large_kernel = arg.large_kernel,
                          num_nodes=torch_dataset.n_nodes,
                          windows=arg.window,
                          horizon=arg.horizon,
                          rw_sample=arg.rw_sample,
                          rw_length=arg.rw_length,
                          dropout=arg.dropout)
    
    logger = TensorBoardLogger(
            save_dir=f"logs/{dataset.name}",
            name=get_model_log_name(model,torch_dataset)
    )

    predictor = Predictor(
        model=model,                   # our initialized model
        optim_class=torch.optim.Adam,  # specify optimizer to be used...
        optim_kwargs={'lr': arg.learning_rate,
                      'weight_decay':arg.weight_decay
                      },
        loss_fn=loss_fn,               # which loss function to be used
        metrics=log_metrics,                # metrics to be logged during train/val/test
        scale_target = False,
        # scheduler_class = MultiStepLR,
        # scheduler_kwargs = {'milestones':[10, 30, 60]}
    )

    checkpoint_callback = ModelCheckpoint(
    dirpath=f'model_checkpoint/{dataset.name}/{model.__class__.__name__}',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
    verbose=True,
    )

    early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=arg.patience,
            mode='min',
            min_delta = 0.001
        )

    trainer = Trainer(
            max_epochs=arg.train_epochs,
            limit_train_batches = arg.limit_train_batches,
            accelerator=arg.accelerator,
            num_sanity_val_steps=0,
            devices=[arg.device],
            gradient_clip_val=5,
            callbacks=[early_stop_callback],
            # default_root_dir="logs",
            check_val_every_n_epoch = 5,
            logger=logger    
    )
    trainer.fit(predictor, datamodule=dm)

    predictor.freeze()

    trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    pass
