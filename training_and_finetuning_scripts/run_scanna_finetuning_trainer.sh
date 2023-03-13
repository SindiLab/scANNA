# These are the recommended parameters for training scANNA.
# Please make sure to change the appropriate machine-specific parameters 
# before running this bash script.

CUDA_VISIBLE_DEVICES=0 ~/data/miniconda3/bin/python /home/aheydari/scANNA_Project/scANNA_Mains/train_or_finetune_scanna.py --dataset "LinEtAl_Finetune" --batch_size 512 --lr 0.0001 --optimizer "adam" --epochs 50 --data_path "/home/aheydari/data/NACT_Data/TransferLearning/ExactIntersectingGenes_PDAC/ExactGenes_LinEtAl_5k_raw_transferlearning_transferTO.h5ad" --annotation_key "celltype" --branches 4 --workers 28 --lr_decay --decay_epoch 25 --decay_frequency 5 --decay_reset_epoch 101 --where_to_save_model "/home/aheydari/data/scANNA_Trained_Models/" --pretrained "/home/aheydari/data/scANNA_Trained_Models/pretrained_for_finetuning.pth"
