import os
import random
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything as pl_seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from tsl.datasets import MetrLA, AirQuality, PemsBay, PvUS, LargeST
from tsl.engines import Predictor, Imputer
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter, AtTimeStepSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.metrics import torch as torch_metrics
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from nn.forecasting import ModernST
from nn.imputation import ModernSTImpute
from utils import str2bool, MaskedRMSE
from dataset_utils import SDWPE, HO_Pre, HO_Imp


def get_parser():
    """Configure command line arguments"""
    parser = argparse.ArgumentParser(description='ModernST Spatial-Temporal Forecasting/Imputation')

    # Task configuration
    parser.add_argument('--task', type=str, default='imputation', choices=['forecasting', 'imputation'],
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
    parser.add_argument('--alpha', type=float, default=0.5, 
                       help='Alpha hyperparameter for biased random walk (0-1)')
    parser.add_argument('--beta', type=float, default=0.5, 
                       help='Beta hyperparameter for biased random walk (0-1)')
    
    # Coordinate configuration
    parser.add_argument('--coord_type', type=str, default='relative', 
                       choices=['relative', 'geographic'],
                       help='Coordinate type for spatial features')
    parser.add_argument('--use_delaunay', type=str2bool, default=True, 
                       help='Use Delaunay triangulation for triangle features')

    # ModernST model configuration
    parser.add_argument('--input_size', type=int, default=1, help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--exog_size', type=int, default=4, help='Exogenous feature dimension')
    parser.add_argument('--ff_size', type=int, default=128, help='Feed-forward layer size')
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[5, 3], 
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

    # Training configuration
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--train_epochs', type=int, default=30, help='Maximum training epochs')
    parser.add_argument('--limit_train_batches', type=int, default=100, 
                       help='Limit training batches per epoch (for debugging)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--scale_target', type=str2bool, default=False, help='Scale target for loss computation')

    # Hardware configuration
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu'],
                       help='Training accelerator')
    parser.add_argument('--devices', type=int, default=0, help='GPU device ID')

    # Optuna configuration
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--study_name', type=str, default='modernst_tuning', help='Study name')

    # Add preset argument
    parser.add_argument('--tune_preset', type=str, default='minimal',
                       choices=['minimal', 'architecture', 'random_walk', 'full', 'custom'],
                       help='Preset tuning configuration')

    return parser


class ModernSTHyperparameterTuner:
    """Hyperparameter tuning for ModernST using Optuna - supports both forecasting and imputation"""
    
    def __init__(self, args):
        self.args = args
        self.best_trial_results = {}
        
        # Define tunable hyperparameters configuration
        # Set tune_param=True to enable tuning for that parameter
        self.tunable_params = {
            # Optimization hyperparameters
            'learning_rate': {
                'tune': True,
                'type': 'float',
                'range': (1e-5, 1e-2),
                'log': True
            },
            # Model architecture  
            'hidden_size': {
                'tune': True,
                'type': 'categorical',
                'choices': [32, 64]
            },
            
            # Random walk parameters
            'rw_samples': {
                'tune': False,  # Disabled as requested
                'type': 'int',
                'range': (50, 100),
                'step': 10
            },
            'rw_length': {
                'tune': False,  # Disabled as requested
                'type': 'int', 
                'range': (3, 10)
            },
            
            # Higher-order adjacency parameters
            'alpha': {
                'tune': False,  # Disabled as requested
                'type': 'categorical',
                'choices': [0.5,0.6,0.7,0.8,0.9],
                'condition': 'bias_walk'  # Only tune if bias_walk is True
            },
            'beta': {
                'tune': False,  # Disabled as requested
                'type': 'categorical', 
                'choices': [0.5,0.6,0.7,0.8,0.9],
                'condition': 'bias_walk'  # Only tune if bias_walk is True
            },
        }
        
    def setup_seed(self, seed):
        """Set random seeds for reproducibility"""
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
            torch.backends.cudnn.benchmark = True
            
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def get_dataset(self, dataset_name, task='forecasting', p_fault=0.0, p_noise=0.0, min_seq=12, max_seq=48):
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
        if task == 'imputation' and (p_fault > 0 or p_noise > 0):
            dataset = add_missing_values(
                dataset,
                p_fault=p_fault,
                p_noise=p_noise,
                min_seq=min_seq,
                max_seq=max_seq,
                seed=42  # Fixed seed for reproducibility
            )
        
        return dataset

    def get_coordinates(self, dataset_name):
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
            return None

    def get_data_splitter(self, dataset_name):
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
    
    def create_dataset(self, dataset, connectivity, covariates):
        """Create appropriate dataset based on task type"""
        
        if self.args.task == 'forecasting':
            return HO_Pre(
                target=dataset.dataframe(),
                connectivity=connectivity,
                mask=dataset.mask,
                covariates=covariates,
                window=self.args.window,
                horizon=self.args.horizon,
                order=self.args.order,
                diagonal=self.args.diagonal,
                bias=self.args.bias_walk,
                norm=self.args.norm,
                alpha=self.args.alpha,
                beta=self.args.beta,
                points=self.get_coordinates(self.args.data),
                coord_type=self.args.coord_type,
                use_delaunay=self.args.use_delaunay
            )
        
        elif self.args.task == 'imputation':
            # For imputation, use training_mask for training and eval_mask for evaluation
            training_mask = getattr(dataset, 'training_mask', dataset.mask)
            eval_mask = getattr(dataset, 'eval_mask', ~dataset.mask)
            
            return HO_Imp(
                target=dataset.dataframe(),
                eval_mask=eval_mask,
                mask=training_mask,
                connectivity=connectivity,
                covariates=covariates,
                window=self.args.window,
                order=self.args.order,
                diagonal=self.args.diagonal,
                bias=self.args.bias_walk,
                norm=self.args.norm,
                alpha=self.args.alpha,
                beta=self.args.beta,
                points=self.get_coordinates(self.args.data),
                coord_type=self.args.coord_type,
                use_delaunay=self.args.use_delaunay,
                transform=MaskInput()  # Important for imputation
            )
        
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
    
    def setup_data_module(self):
        """Setup data module that will be reused across trials"""
        dataset = self.get_dataset(
            self.args.data, 
            task=self.args.task,
            p_fault=self.args.p_fault,
            p_noise=self.args.p_noise,
            min_seq=self.args.min_seq,
            max_seq=self.args.max_seq
        )
        
        connectivity = dataset.get_connectivity(
            threshold=self.args.threshold,
            include_self=False,
            normalize_axis=1,
            force_symmetric=(not self.args.directed),
            layout="edge_index"
        )
        
        covariates = {'u': dataset.datetime_encoded('day').values}
        
        torch_dataset = self.create_dataset(dataset, connectivity, covariates)
        
        scalers = {'target': StandardScaler(axis=(0, 1))}
        
        datamodule = SpatioTemporalDataModule(
            dataset=torch_dataset,
            scalers=scalers,
            splitter=self.get_data_splitter(self.args.data),
            batch_size=self.args.batch_size,
            workers=self.args.num_workers
        )
        datamodule.setup()
        
        return datamodule, torch_dataset.n_nodes
        
    def set_tuning_config(self, **kwargs):
        """
        Dynamically enable/disable tuning for specific parameters
        
        Usage:
            tuner.set_tuning_config(
                ff_size=True,       # Enable ff_size tuning
                dropout=True,       # Enable dropout tuning  
                rw_samples=False    # Disable rw_samples tuning
            )
        """
        for param_name, tune_flag in kwargs.items():
            if param_name in self.tunable_params:
                self.tunable_params[param_name]['tune'] = tune_flag
                print(f"Set {param_name} tuning: {'ON' if tune_flag else 'OFF'}")
            else:
                print(f"Warning: Parameter '{param_name}' not found in tunable_params")
    
    def suggest_hyperparameters(self, trial):
        """Define hyperparameter search space based on configuration"""
        hyperparams = {}
        
        for param_name, config in self.tunable_params.items():
            # Skip if tuning is disabled
            if not config.get('tune', False):
                continue
                
            # Skip if task-specific and doesn't match current task
            if 'task' in config and config['task'] != self.args.task:
                continue
                
            # Skip if conditional parameter and condition not met
            if 'condition' in config:
                if config['condition'] == 'bias_walk' and not getattr(self.args, 'bias_walk', False):
                    continue
            
            # Suggest parameter based on type
            if config['type'] == 'float':
                if config.get('log', False):
                    value = trial.suggest_float(param_name, *config['range'], log=True)
                else:
                    value = trial.suggest_float(param_name, *config['range'])
                    
            elif config['type'] == 'int':
                step = config.get('step', 1)
                value = trial.suggest_int(param_name, *config['range'], step=step)
                
            elif config['type'] == 'categorical':
                value = trial.suggest_categorical(param_name, config['choices'])
            
            hyperparams[param_name] = value
        
        # Handle special case: kernel_sizes (depends on num_blocks)
        if 'num_blocks' in hyperparams:
            kernel_sizes = []
            for i in range(hyperparams['num_blocks']):
                kernel_size = trial.suggest_int(f'kernel_size_{i}', 3, 7, step=2)
                kernel_sizes.append(kernel_size)
            hyperparams['kernel_sizes'] = kernel_sizes
        
        return hyperparams
    
    def get_param_value(self, param_name, hyperparams):
        """Get parameter value from hyperparams or fallback to args"""
        if param_name in hyperparams:
            return hyperparams[param_name]
        else:
            return getattr(self.args, param_name)
    
    def create_model(self, hyperparams, num_nodes):
        """Create appropriate model based on task type"""
        
        if self.args.task == 'forecasting':
            return ModernST(
                input_size=self.args.input_size,
                hidden_size=self.get_param_value('hidden_size', hyperparams),
                exog_size=self.args.exog_size,
                ff_size=self.get_param_value('ff_size', hyperparams),
                num_nodes=num_nodes,
                kernel_sizes=self.get_param_value('kernel_sizes', hyperparams),
                spatial_step=self.args.spatial_step,
                patch_size=(self.get_param_value('rw_samples', hyperparams), 2),
                horizon=self.args.horizon,
                rw_length=self.get_param_value('rw_length', hyperparams),
                rw_samples=self.get_param_value('rw_samples', hyperparams),
                bias_walk=self.args.bias_walk,
                dropout=self.get_param_value('dropout', hyperparams),
                activation=self.args.activation,
                use_learned_adj=self.args.use_learned_adj
            )
        
        elif self.args.task == 'imputation':
            return ModernSTImpute(
                input_size=self.args.input_size,
                hidden_size=self.get_param_value('hidden_size', hyperparams),
                exog_size=self.args.exog_size,
                ff_size=self.get_param_value('ff_size', hyperparams),
                n_nodes=num_nodes,
                kernel_sizes=self.get_param_value('kernel_sizes', hyperparams),
                spatial_step=self.args.spatial_step,
                patch_size=(self.get_param_value('rw_samples', hyperparams), 2),
                rw_length=self.get_param_value('rw_length', hyperparams),
                rw_samples=self.get_param_value('rw_samples', hyperparams),
                bias_walk=self.args.bias_walk,
                dropout=self.get_param_value('dropout', hyperparams),
                activation=self.args.activation,
                use_learned_adj=self.args.use_learned_adj
            )
        
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
    
    def create_engine(self, model, hyperparams):
        """Create appropriate engine (Predictor or Imputer) based on task type"""
        
        # Common loss function
        loss_fn = torch_metrics.MaskedMAE()
        
        if self.args.task == 'forecasting':
            metrics = {
                'mae': torch_metrics.MaskedMAE(),
                'rmse': MaskedRMSE(),
            }
            
            return Predictor(
                model=model,
                optim_class=torch.optim.Adam,
                optim_kwargs={
                    'lr': self.get_param_value('learning_rate', hyperparams),
                    'weight_decay': self.get_param_value('weight_decay', hyperparams)
                },
                loss_fn=loss_fn,
                metrics=metrics,
                scale_target=False
            )
        
        elif self.args.task == 'imputation':
            metrics = {
                'mae': torch_metrics.MaskedMAE(),
                'mse': torch_metrics.MaskedMSE(),
                'mre': torch_metrics.MaskedMRE(),
                'mape': torch_metrics.MaskedMAPE()
            }
            
            return Imputer(
                model=model,
                optim_class=torch.optim.Adam,
                optim_kwargs={
                    'lr': self.get_param_value('learning_rate', hyperparams),
                    'weight_decay': self.get_param_value('weight_decay', hyperparams)
                },
                loss_fn=loss_fn,
                metrics=metrics,
                scale_target=False,
                whiten_prob=self.get_param_value('whiten_prob', hyperparams)
            )
        
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
    
    def objective(self, trial):
        """Optuna objective function"""
        # Set seed for reproducibility
        self.setup_seed(self.args.random_seed)
        
        # Get hyperparameters for this trial
        hyperparams = self.suggest_hyperparameters(trial)
        
        # Setup data (reuse cached if available)
        if not hasattr(self, '_cached_dm'):
            self._cached_dm, self._cached_num_nodes = self.setup_data_module()
        
        datamodule = self._cached_dm
        num_nodes = self._cached_num_nodes
        
        # Create model with suggested hyperparameters
        model = self.create_model(hyperparams, num_nodes)
        
        # Create engine (Predictor or Imputer)
        engine = self.create_engine(model, hyperparams)
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_mae',
            patience=self.args.patience,
            mode='min',
            min_delta=0.001
        )
        callbacks.append(early_stop_callback)
        
        # Pruning callback for Optuna
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_mae")
        callbacks.append(pruning_callback)
        
        # Configure trainer
        trainer = Trainer(
            max_epochs=self.args.train_epochs,
            limit_train_batches=self.args.limit_train_batches,
            accelerator=self.args.accelerator,
            devices=[self.args.devices],
            gradient_clip_val=5.0,
            # callbacks=callbacks,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=3,
            enable_progress_bar=False,  # Reduce noise during tuning
            logger=False,  # Disable logging for trials
        )
        
        try:
            # Train model
            trainer.fit(engine, datamodule=datamodule)
            
            # Test model
            engine.freeze()
            trainer.test(ckpt_path="best", dataloaders=datamodule.test_dataloader())
            
            # Get test results
            test_mae = trainer.callback_metrics["test_mae"].item()
            test_rmse = trainer.callback_metrics.get("test_rmse", torch.tensor(float('inf'))).item()
            
            # Log trial results
            self.log_trial_result(trial, hyperparams, test_mae, test_rmse)
            
            return test_mae
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return float('inf')
    
    def log_trial_result(self, trial, hyperparams, test_mae, test_rmse):
        """Log individual trial results"""
        results_dir = f'optuna_results/{self.args.data}/{self.args.task}/{self.args.study_name}'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f'{results_dir}/trial_results.txt', 'a') as f:
            f.write(f"Trial {trial.number}:\n")
            f.write(f"  Test MAE: {test_mae:.4f}\n")
            f.write(f"  Test RMSE: {test_rmse:.4f}\n")
            f.write(f"  Hyperparameters:\n")
            for key, value in hyperparams.items():
                f.write(f"    {key}: {value}\n")
            f.write("-" * 60 + "\n")
    
    def run_study(self):
        """Run the complete hyperparameter optimization study"""
        print(f"=== ModernST {self.args.task.capitalize()} Hyperparameter Tuning ===")
        print(f"Dataset: {self.args.data.upper()}")
        print(f"Task: {self.args.task}")
        print(f"Study: {self.args.study_name}")
        print(f"Trials: {self.args.n_trials}")
        print(f"Random seed: {self.args.random_seed}")
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{self.args.study_name}_{self.args.task}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.args.random_seed)
        )
        
        # Initialize results file
        results_dir = f'optuna_results/{self.args.data}/{self.args.task}/{self.args.study_name}'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f'{results_dir}/trial_results.txt', 'w') as f:
            f.write(f"OPTUNA HYPERPARAMETER TUNING - {self.args.data.upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Task: {self.args.task}\n")
            f.write(f"Study: {self.args.study_name}\n")
            f.write(f"Dataset: {self.args.data}\n")
            f.write(f"Total trials: {self.args.n_trials}\n")
            if self.args.task == 'imputation':
                f.write(f"Missing value pattern: p_fault={self.args.p_fault}, p_noise={self.args.p_noise}\n")
            f.write("=" * 60 + "\n\n")
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.args.n_trials, n_jobs = 2)
        
        # Save final results
        self.save_final_results(study, results_dir)
        
        return study.best_trial.params
    
    def save_final_results(self, study, results_dir):
        """Save final study results and analysis"""
        best_trial = study.best_trial
        
        # Summary file
        with open(f'{results_dir}/best_results.txt', 'w') as f:
            f.write("BEST TRIAL RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Task: {self.args.task}\n")
            f.write(f"Best Test MAE: {best_trial.value:.4f}\n")
            f.write(f"Trial Number: {best_trial.number}\n")
            f.write(f"Total Trials Completed: {len(study.trials)}\n")
            f.write(f"Total Trials Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            f.write("\nBest Hyperparameters:\n")
            f.write("-" * 30 + "\n")
            for key, value in best_trial.params.items():
                f.write(f"{key:20}: {value}\n")
        
        # Parameter importance analysis
        try:
            importance = optuna.importance.get_param_importances(study)
            with open(f'{results_dir}/parameter_importance.txt', 'w') as f:
                f.write("PARAMETER IMPORTANCE ANALYSIS\n")
                f.write("=" * 60 + "\n")
                for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{param:20}: {score:.4f}\n")
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
        
        print(f"\nOptimization completed!")
        print(f"Best MAE: {best_trial.value:.4f}")
        print(f"Results saved to: {results_dir}")
    
    def print_tuning_status(self):
        """Print current tuning configuration"""
        print("\n=== CURRENT TUNING CONFIGURATION ===")
        
        tuned_params = []
        fixed_params = []
        
        for param_name, config in self.tunable_params.items():
            # Check if parameter applies to current task
            if 'task' in config and config['task'] != self.args.task:
                continue
                
            # Check conditional parameters
            if 'condition' in config:
                if config['condition'] == 'bias_walk' and not getattr(self.args, 'bias_walk', False):
                    continue
            
            if config.get('tune', False):
                tuned_params.append(param_name)
            else:
                fixed_params.append(param_name)
        
        print(f"TUNED parameters ({len(tuned_params)}): {', '.join(tuned_params)}")
        print(f"FIXED parameters ({len(fixed_params)}): {', '.join(fixed_params)}")
        print("=" * 40)


def quick_tune_setup(tuner, preset='minimal'):
    """
    Quick setup presets for common tuning scenarios
    """
    if preset == 'minimal':
        # Only tune learning rate and weight decay (fastest)
        tuner.set_tuning_config(
            learning_rate=True,
            hidden_size=True,
            rw_samples=False,
            rw_length=False,
            alpha=True,
            beta=True
        )
    
    elif preset == 'architecture':
        # Tune architecture parameters
        tuner.set_tuning_config(
            learning_rate=True,
            hidden_size=True,
            rw_samples=False,
            rw_length=False,
            alpha=False,
            beta=False
        )
    
    elif preset == 'random_walk':
        # Focus on random walk parameters
        tuner.set_tuning_config(
            learning_rate=True,
            hidden_size=False,
            rw_samples=True,
            rw_length=True,
            alpha=False,
            beta=False
        )

    
    
    elif preset == 'full':
        # Tune everything available
        tuner.set_tuning_config(
            learning_rate=True,
            hidden_size=True,
            rw_samples=True,
            rw_length=True,
            alpha=True,
            beta=True
        )
    elif preset == 'custom':
        # Tune everything available
        tuner.set_tuning_config(
            learning_rate=False,
            hidden_size=True,
            rw_samples=True,
            rw_length=True,
            alpha=True,
            beta=True
        )


def main():
    """Main function with enhanced tuning control"""
    parser = get_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'imputation' and args.horizon != args.window:
        print(f"Info: For imputation, setting horizon to window ({args.window})")
        args.horizon = args.window
    
    # Initialize tuner
    tuner = ModernSTHyperparameterTuner(args)
    
    # Apply preset tuning configuration
    quick_tune_setup(tuner, args.tune_preset)
    
    # Example: Custom tuning configuration (uncomment to use)
    # tuner.set_tuning_config(
    #     hidden_size=True,        # Enable hidden_size tuning
    #     rw_samples=True,         # Enable rw_samples tuning
    #     learning_rate=True       # Keep learning_rate tuning enabled
    # )
    
    # Print current configuration
    tuner.print_tuning_status()
    
    # Run hyperparameter optimization
    best_params = tuner.run_study()
    
    print(f"\nBest hyperparameters for {args.task}:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()