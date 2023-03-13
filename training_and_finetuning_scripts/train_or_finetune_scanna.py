"""Main script for training scANNA."""

from __future__ import print_function

# std libs
from adabelief_pytorch import AdaBelief
import argparse
import pandas as pd
from scanna.utilities import (evaluate_classifier, save_checkpoint_classifier,
                            init_weights_xavier_uniform, count_parameters,
                            detailed_count_parameters, load_model)
from scanna import scanpy_to_dataloader
from scanna import AdditiveModel, ProjectionAttention, FineTuningModel
import scanpy
import time
import torch
from torch import nn
import torch.utils.data
import torch.nn.parallel
from tqdm import tqdm

# It is a good idea to turn this on for autograd issues.
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(12345)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--scanna_mode",
    type=str,
    default="projection",
    help=("Which scANNA version we want to use,"
          "default = projection"),
)

parser.add_argument(
    "--branches",
    type=int,
    default=8,
    help=("The number of branches when using projection blocks"
          " and attention, default = 8"),
)

parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    help=("number of epochs to train the classifier,"
          " default = 50"),
)
parser.add_argument(
    "--where_to_save_model",
    type=str,
    default="./",
    help=("Directory where the trained model(s) should be saved"
          " to, default='./' "),
)
parser.add_argument(
    "--data_type",
    type=str,
    default="scanpy",
    help="Type of train/test data, default='scanpy'",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="",
    help=("The name of dataset that scANNA is being trained with,"
          "(used in saving the trained model), default=''"),
)
parser.add_argument(
    "--annotation_key",
    type=str,
    default="celltype",
    help=("The key which has label information for training"
          "scANNA (used in saving the trained model),"
          "default='celltype'"),
)
parser.add_argument("--use_raw_x",
                    default=False,
                    action="store_true",
                    help=("Whether to use adata.X (if set to 'False'decay flag,"
                          "or to use adata.raw.X (if set to 'True'). This "
                          "option depends on the type of preprocessing done."
                          "default=False")
                   )
parser.add_argument("--is_there_validation_split",
                    type=bool,
                    default=False,
                    help=("Whether our annoted data also contains a "
                          "'validation' split or just contains the "
                          "'train'/'test' splits. default=False")
                   )
parser.add_argument("--data_path",
                    type=str,
                    default=None,
                    help=("The path to where the dataset is stored,"
                          "default=None"))
parser.add_argument(
    "--metadata_path",
    type=str,
    default=None,
    help=("Path where the metadata is stored"
          "(which will be used for merging with counts),"
          "default=None"),
)
parser.add_argument("--workers",
                    type=int,
                    default=28,
                    help="The Number of worker for the dataloader, default=28")
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help=("The desired batch size for Dataloader,"
          "default = 128"),
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    help="The desired optimizer, default=adam",
)
parser.add_argument("--lr",
                    type=float,
                    default=1e-04,
                    help="learning rate, default=0.0001")
parser.add_argument("--lr_decay",
                    default=False,
                    action="store_true",
                    help="Enables lr decay, default=False")
parser.add_argument("--decay_rate",
                    type=float,
                    default=0.95,
                    help="the decay rate for the lr, default=0.95")
parser.add_argument(
    "--decay_epoch",
    type=int,
    default=10,
    help=("Number of epochs that must be passed before model"
          "starts decaying the learning rate, default=10"),
)
parser.add_argument(
    "--decay_frequency",
    type=int,
    default=25,
    help=("How often should the learning rate decay (after"
          "decay_epoch` many epochs have passed"
          "default=25"),
)
parser.add_argument(
    "--decay_reset_epoch",
    type=int,
    default=20,
    help=("The epoch in which we should set the learning rate"
          "back to the initial LR decayed, default=20"),
)
parser.add_argument("--decay_flag",
                    type=bool,
                    default=False,
                    help="decay flag, default=False")
parser.add_argument("--momentum",
                    default=0.9,
                    type=float,
                    help="Desired momentum value, default=0.9")
parser.add_argument("--clip",
                    type=float,
                    default=100,
                    help="The threshod for gradient clipping, default= 100")
# The options below are useful for transfer learning, where the starting epoch
# may differ from the current epoch stored in the model
parser.add_argument("--pretrained",
                    default=None,
                    type=str,
                    help="path to pretrained model, default=None")
parser.add_argument("--finetune",
                    default=True,
                    type=bool,
                    help="Whether we want to finetune or not, default=True")

parser.add_argument(
    "--start_epoch",
    default=1,
    type=int,
    help="Manual epoch number (useful for TL with restarts)",
)
parser.add_argument(
    "--reset_epochs",
    default=False,
    action="store_true",
    help=("Whether to start training the pretrained model at 0 "
          "or not, default=False"),
)
parser.add_argument("--cuda",
                    default=True,
                    action="store_true",
                    help="Whether enable cuda or not, default = True")
parser.add_argument("--parallel",
                    default=True,
                    action="store_true",
                    help="enables parallel GPU execution, default = True")
parser.add_argument("--manualSeed",
                    type=int,
                    default=0,
                    help="The value to set as the manual seed, default = 0")

# Initializing global variables outside of the module level global
# interpretation
opt = None
model = None

def main():
    """ Main function for training scANNA"""

    # Setting both the model and input options as global variables.
    global opt, model
    opt = parser.parse_args()
    # Flag for parallel GPU usage.
    para_flag = False
    # Determine the device for training.
    print("==> Using GPU (CUDA)")
    # if we are allowed to run things on CUDA
    if opt.cuda and torch.cuda.is_available():
        device = "cuda"
        # checking for multiple
        if torch.cuda.device_count() > 1 and opt.parallel is True:
            print("==> We will try to use", torch.cuda.device_count(),
                  "GPUs for training")
            para_flag = True

    else:
        device = "cpu"
        print("==> Using CPU")
        print("    -> Warning: Using CPUs will yield to slower training"
              "time than GPUs")
    # --------------------- DATA LOADING ---------------------
    print(f" Looking for file named: {opt.data_path}")
    if opt.data_type.lower() == "scanpy":
        print(f"==> Reading Scanpy object for {opt.dataset}: ")
        adata = scanpy.read_h5ad(opt.data_path)
        if opt.metadata_path is not None:
            metadata = pd.read_csv(opt.metadata_path)
            print("    -> Merging metadata with existing ann data")
            try:
                adata.obs = adata.obs.merge(metadata,
                                            left_on="barcodes",
                                            right_on="barcodes",
                                            copy=False,
                                            suffixes=("", "_drop"))
                adata.obs = adata.obs[
                    adata.obs.columns[~adata.obs.columns.str.endswith("_drop")]]
                adata.obs.index = adata.obs["barcodes"]

            except KeyError as _:
                print("    -> Merging on barcode failed, trying to merge on"
                      "index instead")
                adata.obs["barcodes_orig"] = adata.obs.index.tolist()
                adata.obs = adata.obs.merge(metadata,
                                            left_on="barcodes_orig",
                                            right_on="index",
                                            copy=False,
                                            suffixes=("", "_drop"))
                adata.obs = adata.obs[
                    adata.obs.columns[~adata.obs.columns.str.endswith("_drop")]]
                adata.obs.index = adata.obs["index"]
        # Here we are using a homemade utility function to turn scanpy object
        # to torch dataloader
        train_data_loader, valid_data_loader = scanpy_to_dataloader(
            scanpy_object=adata,
            batch_size=opt.batch_size,
            workers=opt.workers,
            verbose=1,
            annotation_key = opt.annotation_key,
            raw_x=opt.use_raw_x)

        # Getting input output information for the network
        num_genes = [
            batch[0].shape[1] for _, batch in enumerate(valid_data_loader, 0)
        ][0]
        # Getting the number of celltypes.
        number_of_classes = len(adata.obs[opt.annotation_key].unique())
        print(f"==> Number of classes {number_of_classes}")
    else:
        raise ValueError(">-< The data type provided is not recognized yet")

    start_time = time.time()
    # --------------------- BUILDING THE MODEL ---------------------
    branching_heads = opt.branches
    print(f"Number of Branching Projections: {branching_heads}")

    if opt.pretrained:
        mode = "Pretrained"
        print(f"==> Loading pre-trained model from {opt.pretrained}")
        # First we need to setup a placeholder models to load in the trained
        # weights into.
        pretrained_model = ProjectionAttention(
                                input_dimension=num_genes,
                                task_module_output_dimension=number_of_classes,
                                number_of_projections=branching_heads,
                                device=device).to(device)

        _, trained_epoch = load_model(pretrained_model, opt.pretrained)
        print(f"    -> Loaded model was trained for {trained_epoch} epochs")
        if opt.finetune:
            # Rewriting mode to reflect finetuning.
            mode = "Finetuning"
            print(f"    -> Setting up fine-tuning model")
            model = FineTuningModel(
                    pretrained_scanna_model=pretrained_model,
                    task_module_output_dimension = number_of_classes,
                    device=device,
                    ).to(device)

        if not opt.reset_epochs:
            print(f"        -> Not resetting the start epoch to 0 ")
            opt.start_epoch = trained_epoch
        else:
            print(f"        -> Strating finetuning from epoch 0 ")
        print("  -><- Loaded from a pre-trained model!")

    else:
        # for testing purposes
        if opt.scanna_mode.lower() == "additive-attn":
            mode = "AdditiveAttn"
            model = AdditiveModel(input_dimension=num_genes,
                                      output_dimension=number_of_classes,
                                      device=device).to(device)

        elif opt.scanna_mode.lower() == "projection":
            mode = "Pojections+Attention"
            model = ProjectionAttention(
                input_dimension=num_genes,
                task_module_output_dimension=number_of_classes,
                number_of_projections=branching_heads,
                dropout=0.0,
                device=device).to(device)
        else:
            print(f"==> selected training mode: {opt.scanna_mode}")
            raise ValueError("scANNA modes are 'additive-attn' or 'projection'."
                             "Please check your entry.")

        print("Initializing *untrained* weights to Xavier Uniform")
        # initilize the weights in our model
        model.apply(init_weights_xavier_uniform)

    # If parallel gpus is enables, we want to load the model on multiple gpus
    # and distribute the data if we want parallel.
    if para_flag:
        model = nn.DataParallel(model)

    # The loss for the task module of scANNA
    criterion = torch.nn.CrossEntropyLoss()

    # --------------------- Optimizer Setting ---------------------
    if opt.optimizer.lower() == "adam":
        print("==> Optimizer: Adam")
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0.000,
                                     amsgrad=False)

    elif opt.optimizer.lower() == "adabelief":
        print("==> Optimizer: AdaBelief")
        # Adabelief is better to be used with Transformer/LSTM1 parameters.
        optimizer = AdaBelief(params=model.parameters(),
                              lr=opt.lr,
                              eps=1e-16,
                              betas=(0.9, 0.999),
                              weight_decay=1.2e-6,
                              weight_decouple=False,
                              rectify=True,
                              fixed_decay=True,
                              amsgrad=False)

    elif opt.optimizer.lower() == "adamax":
        print("==> Optimizer: AdaMax")
        optimizer = torch.optim.Adamax(params=model.parameters(),
                                       lr=opt.lr,
                                       betas=(0.9, 0.999),
                                       eps=1e-08,
                                       weight_decay=0)

    elif opt.optimizer.lower() == "amsgrad":
        print("==> Optimizer: AmsGrad Variant")
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0,
                                     amsgrad=True)
    else:
        raise ValueError(f"The provided optimizer:{opt.optimizer} is not"
                         "available yet. Please use from: {'adam', 'adamax'"
                         "'adabelief', 'amsgrad'}.")

    # --------------------- LR Decay Setting ---------------------
    decay_setting = opt.lr_decay
    if decay_setting:
        print(f"    -> Training with lr decay {opt.decay_rate}")
        print(f"    -> lr decay occurs at every {opt.decay_frequency} Epochs")
        cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=opt.decay_rate)
    else:
        print("    -> Training with fixed lr ")

    # --------------------- Training Loop ---------------------
    print(f"Model: {model}")
    print(detailed_count_parameters(model))
    print(f"Total number of *trainable parameters* : {count_parameters(model)}")

    cur_iter = 0
    batch_idx = None
    for ep in tqdm(range(0, opt.epochs), desc=f"scANNA {mode} Training"):
        batch_losses = []

        for batch_idx, data in enumerate(train_data_loader):
            cur_iter += 1
            features, labels = data
            optimizer.zero_grad()
            output, _, _ = model(features.float().to(device),
                                            training=True)
            loss = criterion(output.squeeze(), labels.to(device))
            batch_losses.append(loss.data.item())

            loss.backward()
            optimizer.step()

            # Logic for decaying the lr:
            if decay_setting and ep >= opt.decay_epoch:
                if (ep % opt.decay_frequency == 0 and cur_iter != 0 and
                        opt.decay_flag is False):

                    opt.decay_flag = True
                    cf_lr_scheduler.step()
                    for param_group in optimizer.param_groups:
                        print(f"\n    -> Decayed lr to -> {param_group['lr']}")

                if ep % opt.decay_reset_epoch == 0 and opt.decay_flag is True:
                    opt.decay_flag = False
                    # for the encoder
                    for param_group in optimizer.param_groups:
                        if param_group["lr"] < 9e-07:
                            param_group["lr"] = opt.lr
                            print("    -> lr too small."
                                  f" Restting lr to -> {param_group['lr']}")

                        # Resetting the learning rate to a fraction of
                        # the initial lr
                        param_group["lr"] = opt.lr / ((ep / 100) + 1)
                        print("    -> Resetting lr to be"
                              f"= {param_group['lr']}")

                # Resetting the flag to allow for future decays
                if ep % opt.decay_frequency != 0:
                    opt.decay_flag = False

        if not ep % 5:
            print(f"[TRAIN] Epoch: {ep} - Iteration: {cur_iter}, "
                  f"loss: {loss.data.item()}")


# ------------------- Evaluating on Validation Loop -------------------
#        valid_scores = [0]
# If validation data is available, you should uncomment this part for
# early stopping and evaluation of the model.
#         batch_losses = []

#         for batch_idx, data in enumerate(valid_data_loader):
#             features, labels = data
#             features = features.float()
#             output, alphas, context = model(features.float().to(device),
#                                             training=False)

#             loss = criterion(output.squeeze(), labels.to(device))
#             batch_losses.append(loss.data.item())

#         _, _, cur_score, _, _ = evaluate_classifier(
#                                                   valid_data_loader,
#                                                   model,
#                                                   classification_report=False)
# -----------------------------------------------------------------------------

    print(f"==> Total training time {time.time() - start_time}")
    save_checkpoint_classifier(
        model,
        opt.epochs,
        0,
        f"scANNA-{mode}-{opt.dataset}-{branching_heads}Branches-",
        dir_path=opt.where_to_save_model)

    _, _, _, _, _ = evaluate_classifier(valid_data_loader,
                                                  model,
                                                  classification_report=True)

if __name__ == "__main__":
    main()
