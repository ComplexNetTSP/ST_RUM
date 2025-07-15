python3 modern_ST_tuner.py --random_seed 42 --data aq --task imputation  --threshold 0.1 --exog_size 5 --tune_preset custom --coord_type geographic --use_delaunay False --scale_target True --weight_decay 0. --dropout 0. --learning_rate 1e-3 --n_trials 30

python3 run_ModernST.py --data aq --task imputation --threshold 0.1 --exog_size 5 --rw_samples 6 --rw_length 6 --directed True --coord_type geographic --use_delaunay False --weight_decay 0. --dropout 0. --bias_walk True --learning_rate 1e-3 --order 2 --random_seed 42
