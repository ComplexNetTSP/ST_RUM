import os
import random
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything as pl_seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.utils.undirected import is_undirected

from tsl.datasets import MetrLA, AirQuality, PemsBay, PvUS, LargeST
from tsl.engines import Predictor, Imputer
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter, AtTimeStepSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch as torch_metrics
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput

from nn.forecasting import ModernST
from nn.imputation import ModernSTImpute
from utils import str2bool, MaskedRMSE, TimingCallback
from dataset_utils import SDWPE, HO_Pre, HO_Imp
from tsl.utils.casting import torch_to_numpy
from tsl.metrics import numpy as numpy_metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def get_parser():
    """Configure command line arguments"""
    parser = argparse.ArgumentParser(description='ModernST Spatial-Temporal Forecasting/Imputation')

    # Task configuration
    parser.add_argument('--task', type=str, default='forecasting', choices=['forecasting', 'imputation'],
                       help='Task type: forecasting or imputation')
    
    # Random seed
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    # Dataset configuration
    parser.add_argument('--data', type=str, default='sdwpe', 
                       choices=['la', 'sdwpe', 'bay', 'aq', 'aq36', 'pv', 'largeST'],
                       help='Dataset name')
    parser.add_argument('--directed', type=str2bool, default=True, 
                       help='Use directed graph (True) or undirected graph (False)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                       help='Threshold for Gaussian kernel distance in connectivity')

    # Missing value generation (for imputation experiments)
    parser.add_argument('--p_fault', type=float, default=0.0015, 
                       help='Probability of fault (block missing) for imputation datasets')
    parser.add_argument('--p_noise', type=float, default=0.05, 
                       help='Probability of noise (point missing) for imputation datasets')
    parser.add_argument('--min_seq', type=int, default=12, 
                       help='Minimum sequence length for missing blocks')
    parser.add_argument('--max_seq', type=int, default=48, 
                       help='Maximum sequence length for missing blocks')

    # Task-specific configuration
    parser.add_argument('--window', type=int, default=12, help='Input sequence length')
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon length (forecasting only)')
    
    # Higher-order adjacency configuration
    parser.add_argument('--order', type=int, default=2, choices=[0, 1, 2],
                       help='Order of adjacency matrix (0: node, 1: node+edge, 2: node+edge+triangle)')
    parser.add_argument('--diagonal', type=str2bool, default=True, 
                       help='Include diagonal elements in adjacency matrix')
    parser.add_argument('--norm', type=str, default='row', choices=['row', 'col', None],
                       help='Normalization for adjacency matrix')
    # Coordinate configuration
    parser.add_argument('--coord_type', type=str, default='relative', 
                       choices=['relative', 'geographic'],
                       help='Coordinate type for spatial features')
    parser.add_argument('--use_delaunay', type=str2bool, default=True, 
                       help='Use Delaunay triangulation for triangle features')

    # ModernST model configuration
    parser.add_argument('--input_size', type=int, default=1, help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--exog_size', type=int, default=4, help='Exogenous feature dimension')
    parser.add_argument('--ff_size', type=int, default=128, help='Feed-forward layer size')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[7, 5, 3], 
                       help='Kernel sizes for backbone blocks')
    parser.add_argument('--spatial_step', type=int, default=2, 
                       help='Diffusion K step')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--use_learned_adj', type=str2bool, default=True,
                       help='Use learned adjacency matrix')
    
    # Random walk configuration
    parser.add_argument('--rw_samples', type=int, default=100, help='Number of random walk samples')
    parser.add_argument('--rw_length', type=int, default=5, help='Random walk length')
    parser.add_argument('--bias_walk', type=str2bool, default=True, help='Use biased random walk')

    # Imputation-specific parameters
    parser.add_argument('--whiten_prob', type=float, default=0.05, 
                       help='Probability of whitening training data (imputation)')
    parser.add_argument('--prediction_loss_weight', type=float, default=1.0,
                       help='Weight for prediction loss in imputation')
    parser.add_argument('--impute_only_missing', type=str2bool, default=False,
                       help='Impute only missing values or full sequence')
    parser.add_argument('--warm_up_steps', type=int, default=0,
                       help='Warm-up steps for imputation')

    # Training configuration
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--train_epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--limit_train_batches', type=int, default=150, 
                       help='Limit training batches per epoch (for debugging)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--scale_target', type=str2bool, default=True, help='Scale target for loss computation')

    # Hardware configuration
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'],
                       help='Training accelerator')
    parser.add_argument('--devices', type=int, default=0, help='GPU device ID')

    return parser


def seed_everything(seed):
    """Set all random seeds for reproducibility"""
    pl_seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    try:
        import dgl
        dgl.seed(seed)
        dgl.random.seed(seed)
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('medium') 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    return seed


def get_dataset(dataset_name, task='forecasting', p_fault=0.0, p_noise=0.0, min_seq=12, max_seq=48):
    """Load dataset by name with optional missing value injection for imputation"""
    
    # Base dataset loading
    if dataset_name == 'la':
        dataset = MetrLA(root='./data/metrla')
    elif dataset_name == 'sdwpe':
        dataset = SDWPE()
    elif dataset_name == 'bay':
        dataset = PemsBay(root='./data/bay', mask_zeros=True)
    elif dataset_name == 'aq':
        dataset = AirQuality(root='./data/aq', impute_nans=True, small=False)
    elif dataset_name == 'aq36':
        dataset = AirQuality(root='./data/aq36', impute_nans=True, small=True)
    elif dataset_name == 'pv':
        dataset = PvUS(zones="west", freq='30T', mask_zeros=False, root="./data/pvus_west")
        dataset.reduce_(dataset.index < datetime.datetime(2006, 7, 1))
    elif dataset_name == 'largeST':
        dataset = LargeST(subset="sd", mask_zeros=False, root="./data/largeST_sd")
    else:
        raise ValueError(f"Dataset {dataset_name} not available. Choose from: ['la', 'sdwpe', 'bay', 'aq', 'aq36', 'pv', 'largeST']")
    
    # Add missing values for imputation tasks
    if task == 'imputation' and (p_fault > 0 or p_noise > 0) and (dataset_name != 'aq'):
        print(f"Adding missing values: p_fault={p_fault}, p_noise={p_noise}")
        dataset = add_missing_values(
            dataset,
            p_fault=p_fault,
            p_noise=p_noise,
            min_seq=min_seq,
            max_seq=max_seq,
            seed=42  # Fixed seed for reproducibility
        )
    
    return dataset


def get_coordinates(dataset_name):
    """Get spatial coordinates for the dataset"""
    coordinate_configs = {
        'sdwpe': {
            'file': "data/sdwpe/sdwpf_turb_location_elevation.csv",
            'columns': ["x", "y"]
        },
        'la': {
            'file': "data/metrla/locations.csv", 
            'columns': ["latitude", "longitude"]
        },
        'aq': {
            'file': 'data/aq/full437.h5',
            'columns': ["latitude", "longitude"],
            'format': 'hdf'
        },
        'bay': {
            'file': "data/bay/locations.csv",
            'columns': ["latitude", "longitude"]
        }
    }
    
    if dataset_name not in coordinate_configs:
        return None
        
    config = coordinate_configs[dataset_name]
    
    try:
        if config.get('format') == 'hdf':
            location = pd.read_hdf(config['file'], 'stations')
        else:
            location = pd.read_csv(config['file'])
        return location[config['columns']]
    except FileNotFoundError:
        print(f"Warning: Coordinate file {config['file']} not found. Using None.")
        return None


def get_data_splitter(dataset_name):
    """Get appropriate data splitter for dataset"""
    temporal_datasets = ['aq', 'aq36', 'la', 'bay', 'sdwpe']
    if dataset_name in temporal_datasets:
        return TemporalSplitter(val_len=0.1, test_len=0.2)
    
    timestamp_datasets = ['pv', 'largeST']
    if dataset_name in timestamp_datasets:
        return AtTimeStepSplitter(
            first_val_ts=(2006, 5, 1),
            last_val_ts=(2006, 5, 31, 6),
            first_test_ts=(2006, 6, 1)
        )
    
    return TemporalSplitter(val_len=0.1, test_len=0.2)


def create_dataset(args, dataset, connectivity, covariates):
    """Create appropriate dataset based on task type"""
    
    if args.task == 'forecasting':
        return HO_Pre(
            target=dataset.dataframe(),
            connectivity=connectivity,
            mask=dataset.mask,
            covariates=covariates,
            window=args.window,
            horizon=args.horizon,
            order=args.order,
            diagonal=args.diagonal,
            bias=args.bias_walk,
            norm=args.norm,
            points=get_coordinates(args.data),
            coord_type=args.coord_type,
            use_delaunay=args.use_delaunay
        )
    
    elif args.task == 'imputation':
        # For imputation, use training_mask for training and eval_mask for evaluation
        training_mask = getattr(dataset, 'training_mask', dataset.mask)
        eval_mask = getattr(dataset, 'eval_mask', ~dataset.mask)
        
        torch_dataset = HO_Imp(
            target=dataset.dataframe(),
            eval_mask=eval_mask,
            mask=training_mask,
            connectivity=connectivity,
            covariates=covariates,
            window=args.window,
            order=args.order,
            diagonal=args.diagonal,
            bias=args.bias_walk,
            norm=args.norm,
            points=get_coordinates(args.data),
            coord_type=args.coord_type,
            use_delaunay=args.use_delaunay,
            transform=MaskInput()  # Important for imputation
        )
        return torch_dataset
    
    else:
        raise ValueError(f"Unknown task: {args.task}")


def create_model(args, num_nodes):
    """Create appropriate model based on task type"""
    
    if args.task == 'forecasting':
        return ModernST(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            exog_size=args.exog_size,
            ff_size=args.ff_size,
            num_nodes=num_nodes,
            kernel_sizes=args.kernel_sizes,
            spatial_step=args.spatial_step,
            patch_size=(args.rw_samples, 2),
            horizon=args.horizon,
            rw_length=args.rw_length,
            rw_samples=args.rw_samples,
            bias_walk=args.bias_walk,
            dropout=args.dropout,
            activation=args.activation,
            use_learned_adj=args.use_learned_adj
        )
    
    elif args.task == 'imputation':
        return ModernSTImpute(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            exog_size=args.exog_size,
            ff_size=args.ff_size,
            n_nodes=num_nodes,
            kernel_sizes=args.kernel_sizes,
            spatial_step=args.spatial_step,
            patch_size=(args.rw_samples, 2),
            rw_length=args.rw_length,
            rw_samples=args.rw_samples,
            bias_walk=args.bias_walk,
            dropout=args.dropout,
            activation=args.activation,
            use_learned_adj=args.use_learned_adj
        )
    
    else:
        raise ValueError(f"Unknown task: {args.task}")


def create_engine(args, model):
    """Create appropriate engine (Predictor or Imputer) based on task type"""
    
    # Common loss and metrics
    loss_fn = torch_metrics.MaskedMAE()

    # loss_fn = torch_metrics.MaskedMSE()
    
    if args.task == 'forecasting':
        metrics = {
            'mae': torch_metrics.MaskedMAE(),
            'rmse': MaskedRMSE()
            # 'mae_step_2': torch_metrics.MaskedMAE(at=2),
            # 'mae_step_6': torch_metrics.MaskedMAE(at=5),
            # 'mae_step_12': torch_metrics.MaskedMAE(at=11),
            # 'rmse_step_2': MaskedRMSE(at=2),
            # 'rmse_step_6': MaskedRMSE(at=5),
            # 'rmse_step_12': MaskedRMSE(at=11)
        }
        
        return Predictor(
            model=model,
            optim_class=torch.optim.AdamW,
            optim_kwargs={
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay
            },
            loss_fn=loss_fn,
            metrics=metrics,
            scale_target=args.scale_target,
            scheduler_class = MultiStepLR,
            scheduler_kwargs={'milestones':[ 25, 50, 100 ], 'gamma':0.1},
        )
    
    elif args.task == 'imputation':
        metrics = {
            'mae': torch_metrics.MaskedMAE(),
            'mre': torch_metrics.MaskedMRE()
        }
        
        return Imputer(
            model=model,
            optim_class=torch.optim.Adam,
            optim_kwargs={
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay
            },
            loss_fn=loss_fn,
            metrics=metrics,
            scale_target=args.scale_target,
            whiten_prob=args.whiten_prob,
            prediction_loss_weight=args.prediction_loss_weight,
            impute_only_missing=args.impute_only_missing,
            warm_up_steps=args.warm_up_steps,
            scheduler_class = MultiStepLR,
            scheduler_kwargs={'milestones':[ 25, 50, 100 ], 'gamma':0.1},
        )
    
    else:
        raise ValueError(f"Unknown task: {args.task}")
    

def setup_logging_and_checkpoints(args, dataset_name, model):
    """Setup logging and checkpoint callbacks"""
    # Create log directory with task info
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"seed{args.random_seed}_{timestamp}"
    checkpoint_dir = f'model_checkpoint/{dataset_name}/{args.task}/{model.__class__.__name__}/{experiment_name}'
    log_dir = f"logs/{dataset_name}/{args.task}/{experiment_name}"


    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=model.__class__.__name__
    )
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
        verbose=True,
        filename='best-{epoch:02d}-{val_mae:.3f}'
    )
    
    return logger, checkpoint_callback


def evaluate_results(args, collated_outputs):
    """Evaluate and print results based on task type"""

    collated_outputs = torch_to_numpy(collated_outputs)
    if args.task == 'imputation':
        # For imputation: y_hat, y, mask, eval_mask
        y_hat = collated_outputs['y_hat']
        y_true = collated_outputs['y'] 
        eval_mask = collated_outputs.get('eval_mask', None)
        
        print(f"\n=== Imputation Results ===")
        print(f"Predictions shape: {y_hat.shape}")
        print(f"Ground truth shape: {y_true.shape}")
        
        if eval_mask is not None:
            print(f"Evaluation mask shape: {eval_mask.shape}")
            print(f"Total evaluation points: {eval_mask.sum().item()}")
            print(f"Evaluation percentage: {eval_mask.mean().item():.2%}")
        
        # Calculate imputation-specific metrics
        if eval_mask is not None:
            mae = numpy_metrics.mae(y_hat, y_true, eval_mask)
            rmse = numpy_metrics.rmse(y_hat, y_true, eval_mask)
            mre = numpy_metrics.mre(y_hat, y_true, eval_mask)
            
            print(f"Final Imputation Metrics:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MRE: {mre:.4f}")
        
    else:
        # For forecasting: y_hat, y, mask
        y_hat = collated_outputs['y_hat']
        y_true = collated_outputs['y']
        mask = collated_outputs.get('mask', None)
        
        print(f"\n=== Forecasting Results ===")
        print(f"Predictions shape: {y_hat.shape}")
        print(f"Ground truth shape: {y_true.shape}")
        
        if mask is not None:
            print(f"Mask shape: {mask.shape}")
            print(f"Valid prediction points: {mask.sum().item()}")
        
        # Calculate forecasting-specific metrics
        
        mae = numpy_metrics.mae(y_hat, y_true, mask)
        rmse = numpy_metrics.rmse(y_hat, y_true, mask)
        
        print(f"Final Forecasting Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")


def experiment(args):
    """Run complete experiment for forecasting or imputation"""
    print(f"=== ModernST {args.task.capitalize()} Experiment ===")
    print(f"Dataset: {args.data}")
    print(f"Task: {args.task}")
    print(f"Random seed: {args.random_seed}")
    print(f"Higher-order: order={args.order}, diagonal={args.diagonal}")
    print(f"Random walk: samples={args.rw_samples}, length={args.rw_length}, bias={args.bias_walk}")
    
    # Set random seed
    seed_everything(args.random_seed)

    # Load dataset
    print("\n--- Loading Dataset ---")
    dataset = get_dataset(
        args.data, 
        task=args.task,
        p_fault=args.p_fault,
        p_noise=args.p_noise,
        min_seq=args.min_seq,
        max_seq=args.max_seq
    )
    print(f"Dataset loaded: {dataset.name}")
    print(f"Shape: {dataset.numpy().shape}")

    # Setup connectivity
    connectivity = dataset.get_connectivity(
        threshold=args.threshold,
        include_self=False,
        force_symmetric=(not args.directed),
        layout="edge_index"
    )
    # Setup covariates
    covariates = {'u': dataset.datetime_encoded('day').values}

    # Create dataset
    print(f"\n--- Creating Higher-Order Dataset ---")
    torch_dataset = create_dataset(args, dataset, connectivity, covariates)
    print(f"Dataset type: {type(torch_dataset).__name__}")
    print(f"Nodes: {torch_dataset.n_nodes}, Channels: {torch_dataset.n_channels}")
    if hasattr(torch_dataset, 'rw_matrices'):
        print(f"Random walk matrices: {list(torch_dataset.rw_matrices.keys())}")
    
    # Setup data scaling and splitting
    scalers = {'target': StandardScaler(axis=(0, 1))}
    
    datamodule = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=get_data_splitter(args.data),
        batch_size=args.batch_size,
        workers=args.num_workers
    )
    datamodule.setup()
    print(f"Data splits - Train: {len(datamodule.trainset)}, Val: {len(datamodule.valset)}, Test: {len(datamodule.testset)}")

    # Create model
    print(f"\n--- Creating Model ---")
    model = create_model(args, torch_dataset.n_nodes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {total_params:,}")

    # Create engine (Predictor or Imputer)
    print(f"\n--- Creating {args.task.capitalize()} Engine ---")
    engine = create_engine(args, model)

    # Setup logging and checkpoints
    logger, checkpoint_callback = setup_logging_and_checkpoints(args, dataset.name, model)
    
    # Setup early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=args.patience,
        mode='min',
        min_delta=0.0001
    )

    time_callback = TimingCallback()

    # Configure trainer
    trainer_kwargs = {
        'max_epochs': args.train_epochs,
        'accelerator': args.accelerator,
        'devices': [args.devices],
        'gradient_clip_val': 5.0,
        'callbacks': [early_stop_callback, checkpoint_callback, time_callback],
        'logger': False,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 2
    }
    
    # Add limit_train_batches if specified
    if args.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = args.limit_train_batches

    trainer = Trainer(**trainer_kwargs)

    # Train model
    print(f"\n--- Training ---")
    trainer.fit(engine, datamodule=datamodule)

    # Test model
    print(f"\n--- Testing ---")
    engine.freeze()
    # trainer.test(ckpt_path="best", dataloaders=datamodule.test_dataloader())
    
    # Generate detailed predictions with proper collation
    print(f"\n--- Generating Predictions ---")
    test_outputs = trainer.predict(engine,
                                    ckpt_path=checkpoint_callback.best_model_path,
                                    dataloaders=datamodule.test_dataloader())
    collated_outputs = engine.collate_prediction_outputs(test_outputs)
    
    # Evaluate and print detailed results
    evaluate_results(args, collated_outputs)
    
    print(f"\n=== Experiment Completed Successfully! ===")
    return engine, trainer, datamodule, collated_outputs


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'imputation' and args.horizon != args.window:
        print(f"Info: For imputation, setting horizon to window ({args.window})")
        args.horizon = args.window
    
    try:
        engine, trainer, datamodule, results = experiment(args)
        print("Experiment finished successfully!")
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise