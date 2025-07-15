#!/usr/bin/env python3
"""
Optimized Sensitivity Analysis for ModernST Random Walk Hyperparameters
"""

import datetime
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
from itertools import product
from tsl.metrics import numpy as numpy_metrics
from tsl.utils.casting import torch_to_numpy
# Import your main experiment function components
from run_ModernST import (
    get_parser, seed_everything, get_dataset, get_coordinates,
    get_data_splitter, create_model, create_engine, 
    setup_logging_and_checkpoints
)
from tsl.data.datamodule import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from dataset_utils import HO_Pre, HO_Imp
from tsl.transforms import MaskInput


def run_single_experiment(args, dataset, connectivity, covariates, params):
    """Run a single experiment with given parameters"""

    seed_everything(args.random_seed)
    
    # Update args with current parameters
    for param, value in params.items():
        setattr(args, param, value)
    
    # Import dataset creation functions
    
    
    # Create dataset (this part changes with parameters)
    if args.task == 'forecasting':
        torch_dataset = HO_Pre(
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
    else:  # imputation
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
            transform=MaskInput()
        )
    
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
    
    # Create model
    model = create_model(args, torch_dataset.n_nodes)
    
    # Create engine
    engine = create_engine(args, model)
    
    # Setup early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=args.patience,
        mode='min',
        min_delta=0.001
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        dirpath=f'sensitive_analysis_checkpoint/{args.data}/{args.task}/bias_{args.bias_walk}_diagonal_{args.diagonal}/{model.__class__.__name__}/{timestamp}'
    )
    
    # Configure trainer
    trainer_kwargs = {
        'max_epochs': args.train_epochs,
        'accelerator': args.accelerator,
        'devices': [args.devices],
        'gradient_clip_val': 5.0,
        'callbacks': [early_stop_callback, checkpoint_callback],
        'logger': False,
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 3,
        'enable_progress_bar': True,  # Disable progress bar for cleaner output
        'enable_model_summary': False
    }
    
    if args.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = args.limit_train_batches
    
    trainer = Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(engine, datamodule=datamodule)
    
    # Test model
    engine.freeze()
    # trainer.test(ckpt_path="best", dataloaders=datamodule.test_dataloader(), verbose=False)
    
    # Generate predictions
    test_outputs = trainer.predict(engine,
                                   ckpt_path=checkpoint_callback.best_model_path,
                                    dataloaders=datamodule.test_dataloader())
    
    collated_outputs = engine.collate_prediction_outputs(test_outputs)
    
    return collated_outputs


class SensitivityAnalyzer:
    """Sensitivity analyzer with data caching"""
    
    def __init__(self, base_args):
        self.base_args = base_args
        self._cached_data = None
    
    def setup_cached_data(self):
        """Setup and cache data that doesn't change across parameter sweeps"""
        if self._cached_data is None:
            print("Loading and caching dataset...")
            
            # Load dataset once
            dataset = get_dataset(
                self.base_args.data, 
                task=self.base_args.task,
                p_fault=self.base_args.p_fault,
                p_noise=self.base_args.p_noise,
                min_seq=self.base_args.min_seq,
                max_seq=self.base_args.max_seq
            )
            
            # Setup connectivity
            connectivity = dataset.get_connectivity(
                threshold=self.base_args.threshold,
                include_self=False,
                force_symmetric=(not self.base_args.directed),
                layout="edge_index"
            )
            
            # Setup covariates
            covariates = {'u': dataset.datetime_encoded('day').values}
            
            self._cached_data = (dataset, connectivity, covariates)
            print("Dataset cached successfully!")
        
        return self._cached_data
    
    def run_analysis(self, param_ranges):
        """Run sensitivity analysis on specified parameters with caching"""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Setup cached data (load once)
        dataset, connectivity, covariates = self.setup_cached_data()
        
        results = []
        total_runs = len(list(product(*param_values)))
        
        print(f"Running {total_runs} combinations...")
        
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            print(f"Run {i+1}/{total_runs}: {params}")
            
            try:
                # Run experiment with cached data
                collated_outputs = run_single_experiment(self.base_args, dataset, connectivity, covariates, params)
                collated_outputs = torch_to_numpy(collated_outputs)
                
                # Calculate MAE from collated outputs
                y_hat = collated_outputs['y_hat']
                y_true = collated_outputs['y']
                
                if self.base_args.task == 'imputation':
                    # For imputation, use eval_mask
                    eval_mask = collated_outputs.get('eval_mask', None)
                    mae = numpy_metrics.mae(y_hat, y_true, eval_mask)
                else:
                    # For forecasting, use mask if available
                    mask = collated_outputs.get('mask', None)
                    mae = numpy_metrics.mae(y_hat, y_true, mask)
                
                val_mae = mae#.item()
                result = params.copy()
                result['val_mae'] = val_mae
                results.append(result)
                print(f"  MAE: {val_mae:.4f}")
                
                
            except Exception as e:
                print(f"  Error: {e}")
                result = params.copy()
                result['val_mae'] = np.nan
                results.append(result)
        
        return results

def plot_3d_surface(results, param_names, title):
    """Create 3D surface plot"""
    df = pd.DataFrame(results)
    pivot = df.pivot(index=param_names[0], columns=param_names[1], values='val_mae')
    fig = go.Figure(go.Surface(
        x=pivot.columns,
        y=pivot.index, 
        z=pivot.values,
        showscale=False
    ))
    
    fig.update_layout(
        autosize=False,
        # title=title,
        scene=dict(
            xaxis_title=param_names[1],
            yaxis_title=param_names[0],
            zaxis_title='MAE'
        ),
        width=800,
        height=800
    )
    
    return fig

def main():
    parser = get_parser()

    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.random_seed)

    print(args)
    
    param_ranges = {
        'rw_samples': [2, 3, 4, 5, 6, 7, 8],  # [2, 3, 4, 5, 6, 7, 8, 9, 10]
        'rw_length': [2, 3, 4, 5, 6, 7, 8]    # [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    param_names = ['rw_samples', 'rw_length']

    output_prefix = f'rw_sensitivity_{args.task}_{args.data}_bias_{args.bias_walk}_diagonal_{args.diagonal}'
    title = f'Random Walk Sensitivity - {args.task.capitalize()} on {args.data.upper()}'
    
    # Create output folder
    output_folder = 'sensitivity_analysis_results'
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize analyzer with caching
    analyzer = SensitivityAnalyzer(args)
    
    # Run analysis
    results = analyzer.run_analysis(param_ranges)
    
    # Save results
    csv_path = os.path.join(output_folder, f'{output_prefix}.csv')
    # pdf_path = os.path.join(output_folder, f'{output_prefix}.pdf')
    html_path = os.path.join(output_folder, f'{output_prefix}.html')
    
    print(f"Interactive plot saved to: {html_path}")
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Create and save plot
    fig = plot_3d_surface(results, param_names, title)
    # fig.write_image(pdf_path)
    fig.write_html(html_path)
    
    # Print best result
    valid_results = [r for r in results if not np.isnan(r['val_mae'])]
    if valid_results:
        best = min(valid_results, key=lambda x: x['val_mae'])
        param1, param2 = param_names
        print(f"\nBest: {param1}={best[param1]}, {param2}={best[param2]}, MAE={best['val_mae']:.4f}")
    
    print(f"Results saved to folder '{output_folder}':")
    print(f"  {csv_path}")
    print(f"  {html_path}")


if __name__ == "__main__":
    main()