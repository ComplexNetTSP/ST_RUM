from utils import seed_everything
from tsl.datasets import AirQuality, MetrLA, PemsBay, PvUS, PeMS07
from dataset_utils.prediction import HO_Pre
from tsl.data.preprocessing import StandardScaler, RobustScaler
from tsl.data import SpatioTemporalDataModule, TemporalSplitter, SpatioTemporalDataset, AtTimeStepSplitter
import tsl.metrics.torch as torch_metrics
from nn.model import ST_RUM_Model
from tsl.engines import Predictor
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch

def run_experiment():
    
    seed_everything(42)
    
    dataset = MetrLA(root='./data/metrla')
    connectivity = dataset.get_connectivity(threshold=0.4,
                                            include_self=False,
                                            # normalize_axis=1,
                                            force_symmetric=False,
                                            layout="edge_index")

    covariates = {'u': dataset.datetime_encoded('day').values}

    torch_dataset = HO_Pre(target=dataset.dataframe(),
                                        connectivity=connectivity,
                                        mask=dataset.mask,
                                        covariates=covariates,
                                        horizon=12,
                                        window=12,
                                        stride=1)
    
    # Normalize data using mean and std computed over time and node dimensions
    scalers = {'target': StandardScaler(axis=(0, 1))}

    # Split data sequentially:
    #   |------------ dataset -----------|
    #   |--- train ---|- val -|-- test --|
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=8,
    )

    dm.setup()
    
    loss_fn = torch_metrics.MaskedMAE()
    # loss_fn = nn.L1Loss()
    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mse': torch_metrics.MaskedMSE(),
        'mae_step_1': torch_metrics.MaskedMAE(at=0),
    'mae_step_2': torch_metrics.MaskedMAE(at=2),
    'mae_step_3': torch_metrics.MaskedMAE(at=4),
    'mae_step_4': torch_metrics.MaskedMAE(at=6)
    }

    model = ST_RUM_Model(input_size=1,exog_size=2, hidden_size = 16, output_size=1,
                        horizon=12, ff_size = 8, dropout = 0.1,n_layers = 1)
    
    predictor = Predictor(
    model=model,                   # our initialized model
    optim_class=torch.optim.Adam,  # specify optimizer to be used...
    optim_kwargs={'lr': 1e-3,
                  'weight_decay':1e-3},
    loss_fn=loss_fn,               # which loss function to be used
    metrics=log_metrics,                # metrics to be logged during train/val/test
    scale_target = False,
    # scheduler_class = MultiStepLR,
    # scheduler_kwargs = {'milestones':[40, 80, 120]}
    )
    
    checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
    )

    early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=30,
            mode='min'
        )



    trainer = Trainer(
            max_epochs=300,
        # limit_train_batches = 32,
        # default_root_dir=cfg.run.dir,
            #logger=exp_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            num_sanity_val_steps=0,
            #devices=1,
            gradient_clip_val=5,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir="logs",
        profiler="pytorch",
        precision="16-mixed"
        
    )
    trainer.fit(predictor, datamodule=dm)
    
    predictor.freeze()

    trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())
    
if __name__ == '__main__':
    run_experiment()
