python3 modern_ST_tuner.py --random_seed 42 --data sdwpe --task imputation --exog_size 5 --tune_preset custom --bias_walk False --dropout 0. --weight_decay 0. --n_trials 30


python3 run_ModernST.py --data sdwpe --task imputation --exog_size 5 --rw_samples 7 --rw_length 3 --bias_walk False --dropout 0. --weight_decay 0. --learning_rate 1e-2 --order 2 --random_seed 42 

