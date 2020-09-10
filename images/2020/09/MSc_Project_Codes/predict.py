import argparse

import matplotlib

matplotlib.use('Agg')
import torch

from data_loader import get_whole_tumour_mask, get_one_hot_3d, get_label4_mask, get_label1_mask
import losses_wolny
import model
import numpy as np
import os, config

# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')

parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--test-file', type=str, default='BraTS19_2013_0_1', metavar='str',
                    help='File name to predict (Should be in the test folder) (default: BraTS19_2013_0_1)')
parser.add_argument('--target-label', type=str, default='wt', metavar='str', choices=['wt', 'et', 'net'],
                    help='Target tumour region (default: whole tumour)')
args = parser.parse_args()

cuda_available = torch.cuda.is_available()
# Test whether CUDA is enabled.
if cuda_available:
    print("CUDA enable. ")
else:
    print("Training on CPU.")

print("Args: {}.".format(parser.parse_args()))

# Model
model = model.Modified3DUNet(4, 1, 16).half()
if cuda_available:
    model.cuda()
assert args.load is not None
model.load_state_dict(torch.load(args.load))

# Loss function
criterion_GDice = losses_wolny.GeneralizedDiceLoss()
criterion_Dice = losses_wolny.DiceLoss()


def predict(fun_get_target_mask, test_file_path):
    """

    :param
        test_file_path: Whole path of an npy file to predict.
            Its shape should be (C x D x W x H).
                Channel(dim0) details: index 0: t1, index1: t1ce, index2: t2, index3:flair, index4: labelled mask
        fun_get_target_mask: a function to get a desired labelled mask
            it can be get_whole_tumour_mask, get_label4_mask, get_label1_mask
    :return: Saved data shape: (C x D x W x H)
    Saved data detail:
        outs: 2 channel output (rounded)
        outs-1channel: 1 channel output (rounded)
        masks: 2 channel mask
        image: 4 channel image
    """
    model.eval()

    with torch.no_grad():
        data_file = np.load(test_file_path)
        data_file_tensor = torch.from_numpy(data_file)
        # Get corresponding image and mask
        image = data_file_tensor[0:4].unsqueeze_(dim=0)  # Make the shape become B x C x W x H x D
        mask = data_file_tensor[4:5].unsqueeze_(dim=0)
        if cuda_available:
            image, mask = image.cuda(), mask.cuda()
        assert image.dim() == 5 and mask.dim() == 5, 'An input should be of shape B x C x W x H x D here'
        print('image size:{}, mask size:{} '.format(image.size(), mask.size()))

        labeled_mask = fun_get_target_mask(mask)
        print('Labelled mask size:{} '.format(labeled_mask.size()))
        image = image.half()
        output = model(image)
        print('Original output size: {}'.format(output.size()))

        # Calculating dice loss using output mask
        output = output.float()

        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        print('Rounded output size: {}'.format(output.size()))

        loss_Dice = criterion_Dice(output, labeled_mask)
        # loss_GDice = criterion_GDice(output, labeled_mask)

        # Calculate loss of this whole 3D image
        print("Predict loss(Dice)(rounded): {}".format(loss_Dice))
        # print("Predict loss(GD)(rounded): {}".format(loss_GDice))

        # Save the segmentation results
        output_folder = os.path.join(config.segmentation_result_folder, args.save)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        np.save(os.path.join(output_folder, 'outs.npy'),
                output.data.float().cpu().numpy())
        np.save(os.path.join(output_folder, 'masks.npy'),
                labeled_mask.data.float().cpu().numpy())
        np.save(os.path.join(output_folder, 'images.npy'),
                image.data.float().cpu().numpy())


"""
    Choosing a proper function for getting desire labelled mask based on the region to predict
    The labelled mask does not affect the prediction. However, it will be used for calculating losses and 
    saved for future comparision. 
"""
assert args.target_label in ['wt', 'et',
                             'net'], 'target label should be among wt (label124) or et (label4) or net (label1)'
get_desired_labelled_mask = None
if args.target_label == 'wt':
    get_desired_labelled_mask = get_whole_tumour_mask
    print("Target mask: whole tumour (label 1 + 2 + 4)")
elif args.target_label == 'et':
    get_desired_labelled_mask = get_label4_mask
    print("Target mask: enhancing tumour (label 4)")
elif args.target_label == 'net':
    get_desired_labelled_mask = get_label1_mask
    print("Target mask: non-enhancing tumour (label 1)")

predict(get_desired_labelled_mask, args.test_file)
