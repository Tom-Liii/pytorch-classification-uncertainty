import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pathlib import Path

import numpy as np
import random
import argparse
import logging
import sys
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
# from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet
from segmentation_models_pytorch.unet.model import Unet
from esd_dataset import ESD_Dataset, get_split, get_id

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import get_device, one_hot_embedding
from losses import relu_evidence, policy_gradient_loss
from metrics import calc_ece_evidence_u, calc_ece_softmax, DiceMetric, calc_mi
from segmentation_models_pytorch.losses.dice import DiceLoss
from esd_dataset import target_img_size
from rl_tuning import evaluation

def predict_single(img_path, model_path):

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--predict", type=int, default=1, help="To perform interface."
    )
    parser.add_argument("--test", action="store_true", help="To test the network.")
    parser.add_argument(
        "--examples", action="store_true", help="To example MNIST data."
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Desired number of train batch size."
    )
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="Desired number of val batch size."
    )
    parser.add_argument(
        "--num_classes", default=5, type=int, help="Desired number of classes."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", type=int, default=0, help="Use uncertainty or not."
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Use uncertainty or not."
    )
    
    parser.add_argument(
        "--mse",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
    parser.add_argument(
        "--digamma",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.",
    )
    parser.add_argument(
        "--log",
        default=0, type=int,
        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.",
    )
    args = parser.parse_args()
    
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    
    seed = args.seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    log_file_name = 'log'
    log_file_name += '_seed_' + str(seed)
    log_file_name += '_batch_' + str(train_batch_size)
    log_file_name += '_classes_' + str(args.num_classes)
    if args.uncertainty :
        log_file_name += '_edl'
        if args.mse:
            log_file_name += '_mse'
        elif args.digamma:
            log_file_name += '_digamma'
        elif args.log:
            log_file_name += '_log'
    # log_file_name = 'debug'
    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('My logger').addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    
    if args.predict:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = args.num_classes

        
        pred_id = get_id(img_path)
        print(f'Path of input image: {pred_id[0]}')
        pred_dataset = ESD_Dataset(pred_id)
        dataloader_pred = DataLoader(pred_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)

    
        # Define the model and optimizer
        model = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=3,
            classes=num_classes,
        )

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Specify the path to the saved checkpoint file
        checkpoint_path = model_path

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the model state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load the optimizer state dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        

        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)
        print('begin to predict')
        # print(f' args: {args}\n model: {model}\n dataloaders: {dataloaders}\n num_classes: {num_classes}\n criterion: {criterion}\n optimizer: {optimizer}\n num_epochs: {num_epochs}\n device: {device}\n uncertainty: {use_uncertainty}')
        make_prediction(
            args, 
            model,
            dataloader_pred,
            num_classes,
            criterion,
            optimizer,
            scheduler=None,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )

def visualize(npy, type=1): 
    import cv2
    # Define the inverse label transformation
    inverse_transform = {
        4: 255,
        0: 212,
        3: 128,
        2: 85,
        1: 42,
    }

    # Convert the label tensor to a mask array
    mask_array = np.zeros_like(npy, dtype=np.uint8)
    for label, value in inverse_transform.items():
        mask_array[npy == label] = value

    # Reshape the mask array to match the desired output size
    target_img_size = 256  # Specify the target image size
    mask_array = cv2.resize(mask_array, (target_img_size, target_img_size), interpolation=cv2.INTER_NEAREST)

    # Save the mask array as a PNG image file
    if type == 1:
        cv2.imwrite('./predict/prediction.png', mask_array)
    else: 
        cv2.imwrite('./predict/label.png', mask_array)


def make_prediction(
    args, 
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
):

    since = time.time()

    if not device:
        device = get_device()
    # print(device)
    logger = logging.getLogger("my logger") 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_dice = 0.0
    losses = {"train": [], 'val':[]}
    accuracy = {"train": [], "val":[]}
    eces = {'train':[], 'val':[]}
    dice_criterion = DiceLoss(
            mode='multiclass',
            classes=num_classes,
            log_loss=False,
            from_logits=True,
            smooth=0.001,
            ignore_index=None,
        )
    dice_metric = DiceMetric()
    

    for epoch in range(num_epochs):
        # print("Epoch {}/{}".format(epoch, num_epochs - 1))
        # print("-" * 10)

    

        model.eval()
        print('Prediction phase')
        phase = 'val'
        running_loss = 0.0
        running_corrects = 0.0
        running_ece = 0.0
        running_dice = 0.0
        running_mi = 0.0
        with torch.no_grad():
            # for i, (inputs, labels) in enumerate(dataloaders[phase]):
            for i, (inputs, labels) in enumerate(dataloaders):
                print(f'Batch {i}:')
                inputs = inputs.to(device)
                labels = labels.to(device)

                if uncertainty:
                    y = one_hot_embedding(labels, num_classes)
                    y = y.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    np.set_printoptions(threshold=np.inf)
                    print('________________')
                    print(f'preds: (type: {type(preds)})')
                    pred_npy = preds.cpu().numpy()
                    pred_npy[pred_npy == 4] = 255
                    pred_npy[pred_npy == 0] = 212
                    pred_npy[pred_npy == 3] = 128
                    pred_npy[pred_npy == 2] = 85
                    pred_npy[pred_npy == 1] = 42
                    print(pred_npy)
                    
                    print(f'labels: (type: {type(labels)})')
                    label_npy = labels.cpu().numpy()
                    label_npy[label_npy == 4] = 255
                    label_npy[label_npy == 0] = 212
                    label_npy[label_npy == 3] = 128
                    label_npy[label_npy == 2] = 85
                    label_npy[label_npy == 1] = 42
                    print(label_npy)
                    print('________________')
                    print('Visualizing... ')
                    visualize(label_npy, 0)
                    visualize(pred_npy, 1)

                    loss = criterion(
                        outputs, y.float(), epoch, num_classes, 10, device
                    )

                    match = torch.eq(preds, labels).float()
                    acc = torch.mean(match)
                    evidence = relu_evidence(outputs)
                    alpha = evidence + 1

                    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                    # print(u.shape)
                    # all_uncertainty.extend(list(u.squeeze().detach().cpu().numpy()))
                    # print(all_uncertainty)
                    expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
                
                    ece = calc_ece_softmax(expected_prob.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    dice = dice_metric.dice_coef(expected_prob, labels)
                    mi = calc_mi(outputs, labels)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match
                    ) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)
                    ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    softmax_pred = F.softmax(outputs, dim=1)

                    # loss = criterion(outputs, labels)
                    loss = dice_criterion(outputs, labels)
                    # dice = _eval_dice(labels.cpu(), preds.detach().cpu(), num_classes)
                    ece = calc_ece_softmax(softmax_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    # dice = 1 - loss.item()
                    dice = dice_metric.dice_coef(softmax_pred, labels)
                    mi = calc_mi(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (torch.sum(preds == labels.data)) / (target_img_size * target_img_size)
                running_ece += ece * inputs.size(0)
                running_dice += dice * inputs.size(0)
                running_mi += mi * inputs.size(0)

       

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)
        epoch_ece = running_ece / len(dataloaders.dataset)
        epoch_dice = running_dice / len(dataloaders.dataset)
        epoch_mi = running_mi / len(dataloaders.dataset)
        
        losses[phase].append(epoch_loss)
        accuracy[phase].append(epoch_acc.item())

        
        
        
        print(
            "{} Epoch:{} loss: {:.4f} dice: {:.4f} ece {:.4f} mi: {:.4f}".format(
                'Test', epoch, epoch_loss, epoch_dice, epoch_ece, epoch_mi
            )
        )
        logger.info('-------------------------------------------')



    

    


if __name__ == "__main__":            
    img_path = 'C:/Users/tom/Desktop/summer_research/ESD_seg/30/image/30_A157_29_3_2016-00_00_00-1.png'
    model_path = "./results/model_uncertainty_mse_batch_4_classes_5_seed_2"
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Inference Time =", current_time)
    predict_single(img_path, model_path)
