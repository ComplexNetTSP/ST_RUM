<<<<<<< HEAD
# ST_RUM
=======
# Modern Structure-Aware Simplicial Spatiotemporal Neural Network

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Download SDWPF dataset from https://figshare.com/articles/dataset/SDWPF_dataset/24798654

Other datasets can be found from Torch-Spatiotemporal Library https://torch-spatiotemporal.readthedocs.io/en/latest/index.html

Please place two csv files for SDWPF in `ST_RUM/data/sdwpe/`

## Run Code

Please find the code in `ST_RUM/script`

**For Tuning in SDWPF dataset:**
```bash
python3 modern_ST_tuner.py --random_seed 42 --data sdwpe --task forecasting --exog_size 4 --tune_preset custom --bias_walk False --n_trials 30
```

**For Running in SDWPF dataset:**
```bash
python3 run_ModernST.py --data sdwpe --task forecasting --exog_size 4 --rw_samples 5 --rw_length 4 --bias_walk False --learning_rate 1e-2 --order 2 --random_seed 42
```
>>>>>>> master
