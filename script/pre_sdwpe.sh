python3 modern_ST_tuner.py --random_seed 42 --data sdwpe --task forecasting --exog_size 4 --tune_preset custom --bias_walk False --n_trials 30

python3 run_ModernST.py --data sdwpe --task forecasting --exog_size 4 --rw_samples 5 --rw_length 4 --bias_walk False --learning_rate 1e-2 --order 2 --random_seed 42
