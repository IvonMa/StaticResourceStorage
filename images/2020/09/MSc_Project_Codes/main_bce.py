# %% -*- coding: utf-8 -*-
"""
    Train a network or Test a given network: BCE version

    Train:
        srun --gres gpu --pty -u python3 -u main_bce.py --train --epoch 100 --batch-size 2 --test-batch-size 1 --optimizer 'ADAM' --lr 0.0008 --lr-decay-frequency 10 --target-label 'wt'

    Test:
        srun --gres gpu --pty python3 main_bce.py --target-label ... --load ...
"""
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import BraTSDatasetUnet, get_whole_tumour_mask, get_label1_mask, get_label4_mask
import losses_wolny
import config, data_loader
import os
import model_bce
from tqdm import tqdm
import utils
import csv
import threading

# %% Training settings
parser = argparse.ArgumentParser(description='3D UNet')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 6)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate at the start time(default: 0.001)')
parser.add_argument('--lr-decay', type=float, default=0.85,
                    help='Decay rate of lr')
parser.add_argument('--lr-decay-frequency', type=float, default=5,
                    help='learning rate at the start time(default: 0.001)')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str',
                    help='Optimizer (default: SGD)')
parser.add_argument('--target-label', type=str, default='wt', metavar='str', choices=['wt', 'et', 'net'],
                    help='Target tumour region (default: whole tumour)')

args = parser.parse_args()  # parse the args actually

# Test whether CUDA is enabled.
cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA enable. ")
else:
    print("Training on CPU.")

print("Args: {}.".format(parser.parse_args()))

"""
    Loading in the Dataset
    Data shape (B x C x W x H x D)
"""

dataset_test = BraTSDatasetUnet(train=False)

test_loader = DataLoader(dataset_test,
                         batch_size=args.test_batch_size,
                         shuffle=False, num_workers=1)

print("Test dataset size:", len(test_loader.dataset))

learning_rate_start = args.lr
learning_rate_final = 0.00001
learning_rate_decay = args.lr_decay
learning_rate_decay_frequency = args.lr_decay_frequency

"""
    Model
"""
model = model_bce.Modified3DUNet(4, 1, 16).half()
if cuda_available:
    model.cuda()

"""
    Loss function
"""
criterion_bce = torch.nn.BCEWithLogitsLoss()
criterion_dice = losses_wolny.DiceLoss()

"""
    Optimizer
"""
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate_start,
                          momentum=0.99)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_start, eps=1e-3)

"""
    Based on the target label, to config some settings
"""
assert args.target_label in ['wt', 'et',
                             'net'], 'target label should be among wt (label124) or et (label4) or net (label1)'

get_desired_labelled_mask = None
current_best_test_loss = 0.2  # Initial value of current best loss based on the target label. (So the loss of a saved model during training should at least lower than it)
if args.target_label == 'wt':
    get_desired_labelled_mask = get_whole_tumour_mask
    current_best_test_loss = 0.18
    print("Target mask: whole tumour (label 1 + 2 + 4)")
elif args.target_label == 'et':
    get_desired_labelled_mask = get_label4_mask
    current_best_test_loss = 0.6
    print("Target mask: enhancing tumour (label 4)")
elif args.target_label == 'net':
    get_desired_labelled_mask = get_label1_mask
    current_best_test_loss = 0.98
    print("Target mask: non-enhancing tumour (label 1)")


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate according to the initial LR and LR decayed every some epochs
    """
    lr = learning_rate_start * (learning_rate_decay ** (epoch // learning_rate_decay_frequency))
    if lr < learning_rate_final:
        lr = learning_rate_final
    print("Current learning rate: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AugmentDataThread(threading.Thread):
    def __init__(self, folder_name):
        super().__init__()
        self.folder_name = folder_name

    def run(self):
        data_loader.generate_augmented_files_randomly(config.preprocessed_training_data_folder, self.folder_name, 80)
        tqdm.write("Data augmentation of this epoch have been finished!")
        # print("Data augmentation of this epoch have been finished!")


def train_a_epoch(epoch):
    """
    The training process of a single epoch
    :param epoch:  For recording and adapting learning rate
    """
    loss_list_epoch = []

    model.train()
    adjust_learning_rate(optimizer, epoch)

    # Generate augment data
    # from train data randomly for the next epoch
    augmented_folder_name_next_epoch = str(epoch + 1)
    AugmentDataThread(augmented_folder_name_next_epoch).start()

    def train_on_a_data_loader(data_loader):
        for batch_idx, (image, mask) in enumerate(tqdm(data_loader)):

            image.requires_grad_(True)
            mask.requires_grad_(True)

            if cuda_available:
                image, mask = image.cuda(), mask.cuda()

            mask = get_desired_labelled_mask(mask)
            image = image.half()
            utils.check_nan_and_inf(batch_idx, image=image, labelled_mask=mask)
            optimizer.zero_grad()

            output = model(image)

            # Calculating dice loss using output mask directly
            output = output.float()
            loss = criterion_bce(output, mask)

            loss.backward()
            optimizer.step()

            loss_list_epoch.append(loss.detach().item())

    # Train on augmented training data
    augmented_folder_name_current_epoch = str(epoch)
    # print("Augmented folder name: {}".format(augmented_folder_name_current_epoch))
    dataset_train_augment = BraTSDatasetUnet(train=True, augment_folder_name=augmented_folder_name_current_epoch)

    train_loader_augment = DataLoader(dataset_train_augment,
                                      batch_size=args.batch_size,
                                      shuffle=True, num_workers=1)
    train_on_a_data_loader(train_loader_augment)

    # Calculating and storing average losses of this epoch
    epoch_average_loss = sum(loss_list_epoch) / len(loss_list_epoch)
    loss_list_global_train.append(epoch_average_loss)

    # Deleting the augmented data of this epoch
    data_loader.delete_augmented_data(augmented_folder_name_current_epoch)

    print('Train Epoch: {} \tAverage Loss(original): {:.6f}'.format(epoch, epoch_average_loss))


def test(epoch=-1):
    """
    Calculating losses on test set
    :param epoch: Just for recording
    """
    loss_list_epoch_dice = []
    loss_list_epoch_bce = []

    model.eval()

    with torch.no_grad():

        loader = test_loader

        for batch_idx, (image, mask) in enumerate(tqdm(loader)):
            if cuda_available:
                image, mask = image.cuda(), mask.cuda()

            mask = get_desired_labelled_mask(mask)
            image = image.half()
            utils.check_nan_and_inf(batch_idx, image=image, labelled_mask=mask)

            output = model(image)

            # Calculating dice loss using output mask directly
            output = output.float()

            loss_bce = criterion_bce(output, mask)

            output_for_dice = output.detach().clone()
            output_for_dice = torch.sigmoid(output_for_dice)
            output_for_dice[output_for_dice >= 0.5] = 1
            output_for_dice[output_for_dice < 0.5] = 0

            loss_dice = criterion_dice(output_for_dice, mask)

            loss_list_epoch_dice.append(loss_dice.detach().item())
            loss_list_epoch_bce.append(loss_bce.detach().item())

            if args.load is not None:
                sensitivity, specificity, hd, precision, dice = utils.calculate_metrics(
                    output_for_dice, mask)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
                if hd != 999:
                    hd_list.append(hd)
                precision_list.append(precision)
                dice_list.append(dice)

        # Average Dice Loss
        # Store losses
        epoch_average_loss_dice = sum(loss_list_epoch_dice) / len(loss_list_epoch_dice)
        epoch_average_loss_bce = sum(loss_list_epoch_bce) / len(loss_list_epoch_bce)

        loss_list_global_test_dice.append(epoch_average_loss_dice)
        loss_list_global_test_bce.append(epoch_average_loss_bce)
        print('Test set: {} \tAverage Loss(Dice): {:.6f}'.format(epoch, epoch_average_loss_dice))
        print('Test set: {} \tAverage Loss(BCE): {:.6f}'.format(epoch, epoch_average_loss_bce))

        if args.load is not None:
            return
        # Save the model if it is a temporarily better one (based on rounded loss)
        # Loss should be lower than the current best and less than 0.2
        global current_best_test_loss
        if epoch_average_loss_dice < current_best_test_loss:
            print("A better model found! The previous best is:{}".format(current_best_test_loss))
            current_best_test_loss = epoch_average_loss_dice
            save_model(epoch, current_best_test_loss)


def save_model(epoch, loss):
    # Draw and save a figure of loss
    plt.cla()  # clear img
    plt.plot(loss_list_global_train, color='green', label='Train Loss (BCE)')

    plt.plot(loss_list_global_test_dice, color='skyblue', label='Test Loss (Dice)')
    plt.plot(loss_list_global_test_bce, color='blue', label='Test Loss (BCE)')
    plt.legend()

    plt.title("{} bs={}, ep={}, lr={}".format(config.output_identifier, args.batch_size,
                                              args.epochs, args.lr))
    plt.xlabel("Number of iterations")
    plt.ylabel("Average loss per epoch")
    plt.savefig(
        os.path.join(config.results_folder,
                     "Fig-{}-target_label={}-bs={}-ep={}-lr={}-loss={}.png".format(config.output_identifier,
                                                                                   args.target_label,
                                                                                   args.batch_size,
                                                                                   epoch,
                                                                                   args.lr, loss)))
    # Save the trained model
    torch.save(model.state_dict(),
               os.path.join(config.results_folder,
                            'Trained_Model-{}-target_label={}-bs={}-epoch={}-lr={}-loss={}'.format(
                                config.output_identifier,
                                args.target_label,
                                args.batch_size,
                                epoch,
                                args.lr, loss)))


loss_list_global_train = []

loss_list_global_test_dice = []
loss_list_global_test_bce = []

# Train a new model
if args.train:
    for i in range(args.epochs):
        print("\nEpoch {} start.".format(i))
        train_a_epoch(i)
        test(i)

    save_model(args.epochs, loss_list_global_test_bce[-1])  # Save the model after training anyway

    # Save loss list
    loss_file_path = os.path.join(config.results_folder,
                                  'Loss-{}-target_label={}-bs={}-lr={}.csv'.format(
                                      config.output_identifier,
                                      args.target_label,
                                      args.batch_size,
                                      args.lr))
    with open(loss_file_path, 'w') as csvfile:
        header = range(len(loss_list_global_train))
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([''] + list(header))
        writer.writerow(["Train Loss (BCE)"] + loss_list_global_train)
        writer.writerow(["Test Loss (BCE)"] + loss_list_global_test_bce)
        writer.writerow(["Test Loss (DICE)"] + loss_list_global_test_dice)

# Test a given model
elif args.load is not None:
    model.load_state_dict(torch.load(args.load))
    # sensitivity, specificity, hd, precision, dice
    sensitivity_list = []
    specificity_list = []
    hd_list = []
    precision_list = []
    dice_list = []

    test()

    print("Loss(Dice, Rounded Output) of Each sample:{}".format(loss_list_global_test_dice))
    utils.print_statistics(dice_list)
    print("Sensitivity average:{}".format(sum(sensitivity_list) / len(sensitivity_list)))
    print("Specificity average:{}".format(sum(specificity_list) / len(specificity_list)))
    print("Hd average:{}".format(sum(hd_list) / len(hd_list)))
    print("Precision:{}".format(sum(precision_list) / len(precision_list)))
    print("Dice:{}".format(sum(dice_list) / len(dice_list)))
