import argparse
import os

import matplotlib
import numpy as np
import torch

import config
import losses_wolny
import model
import csv
import validate
import utils
from data_loader import get_whole_tumour_mask, get_label4_mask, get_label1_mask, \
    get_enhancing_tumour_mask, get_tumour_core_mask

matplotlib.use('Agg')

"""
    Using three trained model (predicting wt, predicting label 4, predicting label 1) to generate an 
    overall prediction and calculate the accuracy based on the requirement of Brats 
    
    srun --gres gpu --pty python3 predict_all_labels.py --load_wt .. --load_et .. --load_net ..  --test-all -m test

"""
# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')

parser.add_argument('--load_wt', type=str, default=None, metavar='str',
                    help='Whole tumour weight file to load (default: None)')
parser.add_argument('--load_et', type=str, default=None, metavar='str',
                    help='Enhancing tumour weight file to load (default: None)')
parser.add_argument('--load_net', type=str, default=None, metavar='str',
                    help='Non-enhancing tumour weight file to load (default: None)')
parser.add_argument('--save', action='store_true', default=False,
                    help='Whether to store the result')
parser.add_argument('--test-file', type=str, default=None, metavar='str',
                    help='File to predict (A full path of it)')
parser.add_argument('--test-all', action='store_true', default=False,
                    help='Test all test dataset files and give an average loss.')
parser.add_argument("-m", "--mode", help="It can be test or val. Should be used along with --test-all",
                    type=str)
args = parser.parse_args()

cuda_available = torch.cuda.is_available()
# Test whether CUDA is enabled.
if cuda_available:
    print("CUDA enable. ")
else:
    print("Training on CPU.")

print("Args: {}.".format(parser.parse_args()))

if args.mode == 'test':
    source_file_directory = config.preprocessed_testing_data_folder
else:
    source_file_directory = config.preprocessed_validation_data_folder


def predict(fun_get_target_mask, model_parameter_file_path, test_file_path):
    # Model
    current_model = model.Modified3DUNet(4, 1, 16).half()
    if cuda_available:
        current_model.cuda()
    assert model_parameter_file_path is not None
    current_model.load_state_dict(torch.load(model_parameter_file_path))

    """

    :param
        test_file_path: Whole path of an npy file to predict.
            Its shape should be (C x D x W x H).
                Channel(dim0) details: index 0: t1, index1: t1ce, index2: t2, index3:flair, index4: labelled mask
        fun_get_target_mask: a function to get a desired labelled mask
            it can be get_whole_tumour_mask, get_label4_mask, get_label1_mask
    :return: Saved data shape: (C x D x W x H)
    Saved data detail:
        outs: 1 channel output (rounded)
        masks: 2 channel mask
        image: 4 channel image
    """
    current_model.eval()

    with torch.no_grad():
        data_file = np.load(test_file_path)
        data_file_tensor = torch.from_numpy(data_file)
        # Get corresponding image and mask
        image = data_file_tensor[0:4].unsqueeze_(dim=0)  # Make the shape become B x C x W x H x D
        mask = data_file_tensor[4:5].unsqueeze_(dim=0)

        if cuda_available:
            image, mask = image.cuda(), mask.cuda()
        assert image.dim() == 5 and mask.dim() == 5, 'An input should be of shape B x C x W x H x D here'

        image = image.half()
        output = current_model(image)

        # Calculating dice loss using output mask
        output = output.float()
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        return output


def predict_brats_label(test_file_path):
    # Segmenting the MR image using the three models
    output_rounded_1channel_wt = predict(get_whole_tumour_mask, args.load_wt, test_file_path)
    output_rounded_1channel_et = predict(get_label4_mask, args.load_et, test_file_path)
    output_rounded_1channel_net = predict(get_label1_mask, args.load_net, test_file_path)

    # Change 3 output masks to their original label
    output_rounded_1channel_wt = output_rounded_1channel_wt * 2
    output_rounded_1channel_et = output_rounded_1channel_et * 4
    output_rounded_1channel_net = output_rounded_1channel_net * 1

    # Combine 3 output masks (priority: et > net > edema)
    final_mask = output_rounded_1channel_et
    final_mask = final_mask + output_rounded_1channel_net
    final_mask[final_mask == 5] = 4  # Set the overlap of et and net mask to et
    final_mask = final_mask + output_rounded_1channel_wt
    final_mask[final_mask == 3] = 1  # Set the overlap of net and wt mask to net
    final_mask[final_mask == 6] = 4  # Set the overlap of et and wt mask to et

    # Get corresponding image and mask (Just for saving and future comparing)
    data_file = np.load(test_file_path)
    data_file_tensor = torch.from_numpy(data_file)
    image = data_file_tensor[0:4].unsqueeze_(dim=0)  # Make the shape become B x C x W x H x D
    labeled_mask = data_file_tensor[4:5].unsqueeze_(dim=0)

    # Calculating Dice Coefficient according to the requirement of Brats
    criterion = losses_wolny.compute_per_channel_dice

    if cuda_available:
        image = image.cuda()
        labeled_mask = labeled_mask.cuda()
        final_mask = final_mask.cuda()

    wt_labelled_mask = get_whole_tumour_mask(labeled_mask)
    wt_predict_mask = get_whole_tumour_mask(final_mask)
    wt_dice_coefficient = criterion(wt_predict_mask, wt_labelled_mask).detach().item()

    et_labelled_mask = get_enhancing_tumour_mask(labeled_mask)
    et_predict_mask = get_enhancing_tumour_mask(final_mask)
    et_dice_coefficient = criterion(et_predict_mask, et_labelled_mask).detach().item()

    tc_labelled_mask = get_tumour_core_mask(labeled_mask)
    tc_predict_mask = get_tumour_core_mask(final_mask)
    tc_dice_coefficient = criterion(tc_predict_mask, tc_labelled_mask).detach().item()
    if args.mode == 'test':
        print("Whole Tumour Dice Coefficient: {}".format(wt_dice_coefficient))
        print("Enhancing Tumour Dice Coefficient: {}".format(et_dice_coefficient))
        print("Tumour Core Dice Coefficient: {}".format(tc_dice_coefficient))

    # Calculating other metrics
    if args.mode == 'test':
        sensitivity_wt, specificity_wt, hd_wt, precision_wt, dice_wt = utils.calculate_metrics(
            wt_predict_mask, wt_labelled_mask)
        wt_sensitivity_list.append(sensitivity_wt)
        wt_specificity_list.append(specificity_wt)
        if hd_wt != 999:
            wt_hd_list.append(hd_wt)
        wt_precision_list.append(precision_wt)
        wt_dice_list.append(dice_wt)

        sensitivity_et, specificity_et, hd_et, precision_et, dice_et = utils.calculate_metrics(
            et_predict_mask, et_labelled_mask)
        et_sensitivity_list.append(sensitivity_et)
        et_specificity_list.append(specificity_et)
        if hd_et != 999:
            et_hd_list.append(hd_et)
        et_precision_list.append(precision_et)
        et_dice_list.append(dice_et)

        sensitivity_tc, specificity_tc, hd_tc, precision_tc, dice_tc = utils.calculate_metrics(
            tc_predict_mask, tc_labelled_mask)
        tc_sensitivity_list.append(sensitivity_tc)
        tc_specificity_list.append(specificity_tc)
        if hd_tc != 999:
            tc_hd_list.append(hd_tc)
        tc_precision_list.append(precision_tc)
        tc_dice_list.append(dice_tc)

    if args.save:
        # Save results
        if not os.path.isdir(config.segmentation_result_folder):
            os.mkdir(config.segmentation_result_folder)
        test_file_name = os.path.basename(test_file_path)
        test_file_name = os.path.splitext(test_file_name)[0]

        np.save(os.path.join(config.segmentation_result_folder, '{}-outs.npy'.format(test_file_name)),
                final_mask.data.float().cpu().numpy())
        np.save(os.path.join(config.segmentation_result_folder, '{}-masks.npy'.format(test_file_name)),
                labeled_mask.data.float().cpu().numpy())
        np.save(os.path.join(config.segmentation_result_folder, '{}-images.npy'.format(test_file_name)),
                image.data.float().cpu().numpy())

        final_mask_3d = final_mask.data.float().squeeze(dim=0).squeeze(dim=0)
        validate.save_pred_mask_nii(config.segmentation_result_folder, final_mask_3d.cpu().numpy(),
                                    int(test_file_name), source_file_directory)
    return wt_dice_coefficient, et_dice_coefficient, tc_dice_coefficient


if args.test_all:
    wt_d_list = []
    et_d_list = []
    tc_d_list = []

    # sensitivity, specificity, hd, precision, dice
    wt_sensitivity_list = []
    et_sensitivity_list = []
    tc_sensitivity_list = []

    wt_specificity_list = []
    et_specificity_list = []
    tc_specificity_list = []

    wt_hd_list = []
    et_hd_list = []
    tc_hd_list = []

    wt_precision_list = []
    et_precision_list = []
    tc_precision_list = []

    wt_dice_list = []
    et_dice_list = []
    tc_dice_list = []

    assert args.mode == 'test' or args.mode == 'val', 'Mode should be test or val'

    patients = os.listdir(source_file_directory)
    for patient in patients:
        if patient.endswith('.npy'):
            print("Segmenting patient:{}".format(patient))
            wt_d, et_d, tc_d = predict_brats_label(os.path.join(source_file_directory, patient))
            wt_d_list.append(wt_d)
            et_d_list.append(et_d)
            tc_d_list.append(tc_d)
    if args.mode == 'val':
        print("All validation data are segmented.")
    else:
        print("All test data are segmented.")
        print("WT Average Dice Coefficient: {}".format(sum(wt_d_list) / len(wt_d_list)))
        print("ET Average Dice Coefficient: {}".format(sum(et_d_list) / len(et_d_list)))
        print("TC Average Dice Coefficient: {}".format(sum(tc_d_list) / len(tc_d_list)))

        # Save metrics
        metrics_file_path = os.path.join(config.segmentation_result_folder, 'metrics.csv')
        with open(metrics_file_path, 'w') as csvfile:
            header = ['wt', 'et', 'tc']
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([''] + list(header))
            writer.writerow(
                ['Dice Coefficient', sum(wt_dice_list) / len(wt_dice_list), sum(et_dice_list) / len(et_dice_list),
                 sum(tc_dice_list) / len(tc_dice_list)])
            writer.writerow(["Dice Coefficient Standard Deviation", utils.calculate_std(wt_dice_list),
                             utils.calculate_std(et_dice_list), utils.calculate_std(tc_dice_list)])
            writer.writerow(["Dice Coefficient Median", utils.calculate_percentile(wt_dice_list, 50),
                             utils.calculate_percentile(et_dice_list, 50),
                             utils.calculate_percentile(tc_dice_list, 50)])
            writer.writerow(["Dice Coefficient 25 percentile", utils.calculate_percentile(wt_dice_list, 25),
                             utils.calculate_percentile(et_dice_list, 25),
                             utils.calculate_percentile(tc_dice_list, 25)])
            writer.writerow(["Dice Coefficient 75 percentile", utils.calculate_percentile(wt_dice_list, 75),
                             utils.calculate_percentile(et_dice_list, 75),
                             utils.calculate_percentile(tc_dice_list, 75)])
            writer.writerow(["Sensitivity average", sum(wt_sensitivity_list) / len(wt_sensitivity_list),
                             sum(et_sensitivity_list) / len(et_sensitivity_list),
                             sum(tc_sensitivity_list) / len(tc_sensitivity_list)])
            writer.writerow(["Specificity average", sum(wt_specificity_list) / len(wt_specificity_list),
                             sum(et_specificity_list) / len(et_specificity_list),
                             sum(tc_specificity_list) / len(tc_specificity_list)])
            writer.writerow(["Hd average", sum(wt_hd_list) / len(wt_hd_list),
                             sum(et_hd_list) / len(et_hd_list),
                             sum(tc_hd_list) / len(tc_hd_list)])
            writer.writerow(["Precision average", sum(wt_precision_list) / len(wt_precision_list),
                             sum(et_precision_list) / len(et_precision_list),
                             sum(tc_precision_list) / len(tc_precision_list)])
else:
    predict_brats_label(args.test_file)
