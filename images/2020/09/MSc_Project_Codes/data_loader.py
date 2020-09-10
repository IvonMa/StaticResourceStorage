import os
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import config


class BraTSDatasetUnet(Dataset):
    __file = []
    dataset_size = 0

    def __init__(self, train=True, augment_folder_name='-1'):

        self.__file = []  # It stores the path of all data

        if train:
            # 1. Reading the file of original data
            original_sample_folder = config.preprocessed_training_data_folder
            original_samples = []
            for file in os.listdir(original_sample_folder):  # "file" is a string
                if file.endswith("npy"):
                    original_samples.append(file)
            # 2. Reading the file of augmented data
            augment_folder = os.path.join(config.augmented_file_cache_folder, augment_folder_name)
            if os.path.isdir(augment_folder):
                augmented_samples = []
                for file in os.listdir(augment_folder):  # "file" is a string
                    if file.endswith("npy"):
                        augmented_samples.append(file)
                # 3. Removing sample which is augmented from the original data set
                not_augmented_samples = []
                for file in original_samples:
                    if file not in augmented_samples:
                        not_augmented_samples.append(file)
                # 4. Combining the original set and the augmented set
                for file in not_augmented_samples:
                    self.__file.append(os.path.join(original_sample_folder, file))
                for file in augmented_samples:
                    self.__file.append(os.path.join(augment_folder, file))
            else:
                for file in original_samples:
                    self.__file.append(os.path.join(original_sample_folder, file))
        else:
            folder = config.preprocessed_testing_data_folder
            for file in os.listdir(folder):  # "file" is a string
                if file.endswith("npy"):
                    self.__file.append(os.path.join(folder, file))

        self.dataset_size = len(self.__file)
        self.__file.sort()
        # print(self.__file)

    def __getitem__(self, index):
        data_file = np.load(self.__file[index])

        return data_file[0:4], data_file[4:5]

    def __len__(self):
        return len(self.__file)


def get_one_hot_2d(mask, num_classes=2):
    """
        Input should be a batch of 2D masks
        Input shape should be: B x W x H or B x 1 x W x H
    """
    mask.squeeze_(dim=1)
    assert mask.dim() == 3
    mask = mask.long()
    one_hot = torch.nn.functional.one_hot(mask, num_classes=num_classes)  # B x W x H x C
    return one_hot.permute(0, 3, 1, 2).contiguous()


def get_one_hot_3d(input_mask, num_classes=2):
    """
        Input should be a batch of 3D masks
        Input shape should be: B x W x H x D or B x 1 x W x H x D
    """
    mask = input_mask.squeeze(dim=1)
    assert mask.dim() == 4
    mask = mask.long()
    # Transfer the label it contains from 0,1,2,4 to 0,1,2,3 when needed
    if num_classes == 4:
        mask[mask == 4] = 3
    one_hot = torch.nn.functional.one_hot(mask, num_classes=num_classes)  # B x W x H x D x C
    return one_hot.permute(0, 4, 1, 2, 3).contiguous()


def get_whole_tumour_mask(input_mask):
    mask = input_mask.detach().clone()
    mask[mask == 2] = 1
    mask[mask == 4] = 1
    return mask


def get_label1_mask(input_mask):
    mask = input_mask.detach().clone()
    mask[mask == 4] = 0
    mask[mask == 2] = 0
    return mask


def get_label4_mask(input_mask):
    mask = input_mask.detach().clone()
    mask[mask == 1] = 0
    mask[mask == 2] = 0
    mask[mask == 4] = 1
    return mask


def get_enhancing_tumour_mask(mask):
    return get_label4_mask(mask)


def get_tumour_core_mask(input_mask):
    mask = input_mask.detach().clone()
    mask[mask == 2] = 0
    mask[mask == 4] = 1
    return mask


def get_desire_mask(target_label, mask):
    assert target_label in ['wt', 'et',
                            'net'], 'target label should be among wt (label124) or et (label4) or net (label1)'
    if target_label == 'wt':
        return get_whole_tumour_mask(mask)
    elif target_label == 'et':
        return get_label4_mask(mask)
    elif target_label == 'net':
        return get_label1_mask(mask)


import shutil
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms import Compose
from multiprocessing import Pool

patch_size = [180, 212, 180]
tr_transforms = []
transform = SpatialTransform_2(
    patch_size, [i // 2 for i in patch_size],
    do_elastic_deform=False,
    do_rotation=True,
    angle_x=(- 20 / 360. * 2 * np.pi, 20 / 360. * 2 * np.pi),
    angle_y=(- 20 / 360. * 2 * np.pi, 20 / 360. * 2 * np.pi),
    angle_z=(- 20 / 360. * 2 * np.pi, 20 / 360. * 2 * np.pi),
    do_scale=True, scale=(1., 1.25),
    border_mode_data='constant', border_cval_data=0,
    border_mode_seg='constant', border_cval_seg=0,
    order_seg=0, order_data=3,
    random_crop=False,
    p_rot_per_sample=0.6, p_scale_per_sample=0.6,
    data_key="data", label_key="seg"
)
tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
tr_transforms.append(transform)
# now we compose these transforms together
tr_transforms = Compose(tr_transforms)

import pre_processing


def augment_data(target_patient, target_folder_name):
    output_folder = os.path.join(config.augmented_file_cache_folder, target_folder_name)
    patient_path = os.path.join(config.preprocessed_training_data_folder, target_patient)
    patient_array = np.load(patient_path)

    patient_array_expanded = give_some_background_space(patient_array)

    # make it bigger in case brain move out after augmented
    img = np.expand_dims(patient_array_expanded[0:4], axis=0)
    seg = np.expand_dims(patient_array_expanded[4:5], axis=0)

    augmented_data = tr_transforms(data=img, seg=seg)
    img = augmented_data["data"].squeeze(axis=0)
    seg = augmented_data["seg"].squeeze(axis=0)

    # It seems there is a bug in Batchgenerators: some background voxel whose intensity is zero will become a decimal
    # which near 0. So we need to put them back to 0
    img[np.abs(img) < 1.e-10] = 0
    seg = np.around(seg)
    # print(img.shape)
    # print(seg.shape)
    all_data = pre_processing.simple_pre(img[0], img[1], img[2], img[3], seg[0])

    # Crop the augmented image in case it is larger than original input image
    all_data_cropped = all_data[:, 0:160, 0:192, 0:160]

    # result = np.concatenate((img, seg), axis=0)
    save_path = os.path.join(output_folder, target_patient)
    # print(all_data.shape)
    np.save(save_path, all_data_cropped)
    # print("finish %s" % target_patient)


def give_some_background_space(patient_array):
    t1_channel = patient_array[0]
    t1ce_channel = patient_array[1]
    t2_channel = patient_array[2]
    flair_channel = patient_array[3]
    seg_channel = patient_array[4]

    t1_channel_expanded = give_some_background_space_one_channel(t1_channel)
    t1ce_channel_expanded = give_some_background_space_one_channel(t1ce_channel)
    t2_channel_expanded = give_some_background_space_one_channel(t2_channel)
    flair_channel_expanded = give_some_background_space_one_channel(flair_channel)
    seg_channel_expanded = give_some_background_space_one_channel(seg_channel)

    all_data = np.zeros([5] + list(t1_channel_expanded.shape), dtype=np.float32)
    all_data[0] = t1_channel_expanded
    all_data[1] = t1ce_channel_expanded
    all_data[2] = t2_channel_expanded
    all_data[3] = flair_channel_expanded
    all_data[4] = seg_channel_expanded

    return all_data


# Input shape: D x W x H
def give_some_background_space_one_channel(patient_array, target_size=None):
    if target_size is None:
        target_size = [180, 212, 180]
    new_size_array = np.zeros(target_size)
    new_size_array[target_size[0] - patient_array.shape[0] - 1:-1, target_size[1] - patient_array.shape[1] - 1:-1,
    target_size[2] - patient_array.shape[2] - 1:-1] = patient_array
    return new_size_array


def run_star(args):
    augment_data(*args)


def generate_augmented_files_randomly(preprocessed_data_folder, folder_name, number_of_augmented_dat=10):
    # delete the previous augmented data in the same folder
    delete_augmented_data(folder_name)
    target_folder = os.path.join(config.augmented_file_cache_folder, folder_name)
    os.mkdir(target_folder)
    # generate new augmented data
    files = os.listdir(preprocessed_data_folder)
    patients = []
    for file in files:
        if file.endswith(".npy"):
            patients.append(file)
    chosen_indexes = random.sample(range(0, len(patients)), number_of_augmented_dat)
    target_patient = []
    for i in chosen_indexes:
        target_patient.append(patients[i])
    p = Pool(8)
    zip_pack = zip(target_patient, [folder_name] * len(patients))
    p.map(run_star, zip_pack)
    p.close()
    p.join()


def delete_augmented_data(folder_name):
    target_folder = os.path.join(config.augmented_file_cache_folder, folder_name)
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)


if __name__ == '__main__':
    generate_augmented_files_randomly(config.preprocessed_training_data_folder, "test_augment", 80)
