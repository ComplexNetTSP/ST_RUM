python3 modern_ST_tuner.py --random_seed 42 --data la --task forecasting --threshold 0.1 --exog_size 4 --tune_preset custom --coord_type geographic --use_delaunay False --n_trials 20

python3 run_ModernST.py --data la --task forecasting --rw_samples 4 --rw_length 2 --coord_type geographic --use_delaunay False --bias_walk True --directed True --order 2 --learning_rate 1e-2 --random_seed 42
