python3 modern_ST_tuner.py --random_seed 42 --data la --task imputation --threshold 0.1 --exog_size 5 --tune_preset custom --coord_type geographic --use_delaunay False --scale_target True --weight_decay 0. --dropout 0. --n_trials 30

python3 run_ModernST.py --data la --task imputation --exog_size 5 --rw_samples 4 --rw_length 8 --coord_type geographic --use_delaunay False --bias_walk True --weight_decay 0. --dropout 0. --learning_rate 1e-2 --order 2 --random_seed 42 
