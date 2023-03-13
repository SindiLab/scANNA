# These are the recommended parameters for training scANNA.
# Please make sure to change the appropriate machine-specific parameters 
# before running this bash script.

CUDA_VISIBLE_DEVICES=0 ~/data/miniconda3/bin/python /home/aheydari/scANNA_Project/scANNA_Package/training_and_finetuning_scripts/train_or_finetune_scanna.py --scanna_mode "projection" --dataset "SCP1361-2Block" --batch_size 512 --lr 0.0001 --optimizer "adam" --epochs 200 --data_path "/home/aheydari/data/NACT_Data/Supervised Benchmarking/SCP1361_qc_hvg_anno_5k_raw_train_split.h5ad" --annotation_key "cluster" --use_raw_x --branches 4 --workers 28 --lr_decay --decay_epoch 25 --decay_frequency 5 --decay_reset_epoch 50 --where_to_save_model "/home/aheydari/data/scANNA_Trained_Models/"
