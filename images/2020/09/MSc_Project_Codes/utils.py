import torch
import numpy as np
import os
import csv
from medpy import metric
import config


def check_nan_and_inf(batch_idx, **kwargs):
    for key in kwargs.keys():
        if torch.isnan(kwargs[key]).any():
            print("Nan found in: {}. batch index:{}".format(key, batch_idx))
        if torch.isinf(kwargs[key]).any():
            print("Inf found in: {}. batch index:{}".format(key, batch_idx))


def calculate_percentile(loss_list, percentile):
    loss_list_np = np.array(loss_list)
    return np.percentile(loss_list_np, percentile)


def calculate_std(loss_list):
    loss_list_np = np.array(loss_list)
    return np.std(loss_list_np)


def print_statistics(loss_list):
    print("Loss standard deviation:{}".format(calculate_std(loss_list)))
    print("Loss Median:{}".format(calculate_percentile(loss_list, 50)))
    print("Loss 25 percentile:{}".format(calculate_percentile(loss_list, 25)))
    print("Loss 75 percentile:{}".format(calculate_percentile(loss_list, 75)))


def calculate_metrics(mask1, mask2):
    mask1 = mask1.detach().cpu().numpy()
    mask2 = mask2.detach().cpu().numpy()

    sensitivity = metric.sensitivity(mask1, mask2)
    specificity = metric.specificity(mask1, mask2)
    if mask1.sum() == 0 or mask2.sum() == 0:
        hd = 999
    else:
        hd = metric.hd95(mask1, mask2)
    precision = metric.precision(mask1, mask2)
    dice = metric.dc(mask1, mask2)
    return sensitivity, specificity, hd, precision, dice


def save_loss_list(target_label, loss_list_global_train, loss_list_global_test, rounded_output_loss_list_global_test):
    # Save loss list
    loss_file_path = os.path.join(config.results_folder,
                                  'Loss-target_label-{}.csv'.format(target_label))
    with open(loss_file_path, 'w') as csvfile:
        header = range(len(loss_list_global_train))
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([''] + list(header))
        writer.writerow(["Train Loss"] + loss_list_global_train)
        writer.writerow(["Test Loss"] + loss_list_global_test)
        writer.writerow(["Test Loss(Rounded)"] + rounded_output_loss_list_global_test)
