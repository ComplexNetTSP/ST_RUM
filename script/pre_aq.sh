python3 modern_ST_tuner.py --random_seed 42 --data aq --task forecasting --threshold 0.1 --exog_size 4 --tune_preset custom --coord_type geographic --use_delaunay False --learning_rate 1e-3 --n_trials 30

python3 run_ModernST.py --data aq --task forecasting --threshold 0.1 --exog_size 4 --rw_samples 8 --rw_length 6 --directed True --coord_type geographic --use_delaunay False --bias_walk True --learning_rate 1e-3 --order 2 --random_seed 42



