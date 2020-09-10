"""
    Some codes refer to the implementation of Isensee et al. https://github.com/MIC-DKFZ/BraTS2017
    Modified by MA Yunfeng
"""
import multiprocessing
import os
import os.path
import os.path as path
import pickle
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

import config

manager = multiprocessing.Manager()
biggest_shape = manager.list([0, 0, 0])


def run_preprocessing_BraTS2017_trainSet(base_folder=config.raw_training_data_folder,
                                         folder_out=config.preprocessed_training_data_folder):
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    ctr = 0
    id_name_conversion = []
    for f in ("HGG", "LGG"):
        fld = os.path.join(base_folder, f)
        patients = os.listdir(fld)
        patients.sort()
        fldrs = [os.path.join(fld, pt) for pt in patients]
        p = Pool(8)
        p.map(run_star, zip(fldrs,
                            [folder_out] * len(patients),
                            range(ctr, ctr + len(patients)),
                            patients))
        p.close()
        p.join()
        for i, j in zip(patients, range(ctr, ctr + len(patients))):
            id_name_conversion.append([i, j, f])
        ctr += (ctr + len(patients))
    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
    print(biggest_shape)


def run_preprocessing_BraTS2017_valOrTestSet(base_folder=config.raw_validation_data_folder,
                                             folder_out=config.preprocessed_validation_data_folder):
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    ctr = 0
    id_name_conversion = []
    fld = os.path.join(base_folder)
    patients = os.listdir(fld)
    patients.sort()
    fldrs = [os.path.join(fld, pt) for pt in patients]
    p = Pool(8)
    p.map(run_star, zip(fldrs,
                        [folder_out] * len(patients),
                        range(ctr, ctr + len(patients)),
                        patients, len(patients) * [False]))
    p.close()
    p.join()
    for i, j in zip(patients, range(ctr, ctr + len(patients))):
        id_name_conversion.append([i, j, 'unknown'])  # not known whether HGG or LGG
    ctr += (ctr + len(patients))
    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")


def run_star(args):
    return run(*args)


def extract_brain_region(image, segmentation, outside_value=0):
    brain_voxels = np.where(segmentation != outside_value)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    return image[resizer], [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]


def cut_off_values_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    return res


def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))

    # Find the biggest shape of all axis
    # Temporary codes start
    for i in range(0, 3):
        if biggest_shape[i] < new_shape[i]:
            biggest_shape[i] = new_shape[i]
    # Temporary codes end

    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0 + int(shape[0]), 0:0 + int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0 + int(shape[0]), 0:0 + int(shape[1]), 0:0 + int(shape[2])] = image
    # return res

    # Crop the image to desire shape just in case
    final_res = res[0:160, 0:192, 0:160]
    return final_res


def n4itk(input_image):
    mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(input_image, mask_image)
    return output


def run(folder, out_folder, id, name, return_if_no_seg=True):
    print(id)
    # Skipping a sample if it lacks any modality or labelled mask
    if not path.isfile(path.join(folder, "%s_flair.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_t1.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_seg.nii.gz" % name)):
        if return_if_no_seg:
            return
    if not path.isfile(path.join(folder, "%s_t1ce.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_t2.nii.gz" % name)):
        return

    # Reading Image and Doing N4ITK bias correction
    t1_img_sitk = sitk.ReadImage(path.join(folder, "%s_t1.nii.gz" % name))
    t1_img_sitk = n4itk(t1_img_sitk)
    t1_img = sitk.GetArrayFromImage(t1_img_sitk).astype(np.float32)

    t1c_img_sitk = sitk.ReadImage(path.join(folder, "%s_t1ce.nii.gz" % name))
    t1c_img_sitk = n4itk(t1c_img_sitk)
    t1c_img = sitk.GetArrayFromImage(t1c_img_sitk).astype(np.float32)

    t2_img_sitk = sitk.ReadImage(path.join(folder, "%s_t2.nii.gz" % name))
    t2_img = sitk.GetArrayFromImage(t2_img_sitk).astype(np.float32)

    flair_img_sitk = sitk.ReadImage(path.join(folder, "%s_flair.nii.gz" % name))
    flair_img = sitk.GetArrayFromImage(flair_img_sitk).astype(np.float32)

    # Reading labelled mask
    try:
        seg_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_seg.nii.gz" % name))).astype(np.float32)
    except RuntimeError:
        seg_img = np.zeros(t1_img.shape)
        # print("Error occurred when reading segmentation file.")
        # return
    except IOError:
        seg_img = np.zeros(t1_img.shape)
        # print("Error occurred when reading segmentation file.")
        # return

    original_shape = t1_img.shape

    brain_mask = (t1_img != t1_img[0, 0, 0]) & (t1c_img != t1c_img[0, 0, 0]) & (t2_img != t2_img[0, 0, 0]) & (
            flair_img != flair_img[0, 0, 0])

    # compute bbox of brain, This is now actually also returned when calling extract_brain_region, but was not at the
    # time this code was initially written. In order to not break anything we will keep it like it was
    brain_voxels = np.where(brain_mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    with open(os.path.join(out_folder, "%03.0d.pkl" % id), 'wb') as f:
        dp = {}
        dp['orig_shp'] = original_shape
        dp['bbox_z'] = [minZidx, maxZidx]
        dp['bbox_x'] = [minXidx, maxXidx]
        dp['bbox_y'] = [minYidx, maxYidx]
        dp['spacing'] = t1_img_sitk.GetSpacing()
        dp['direction'] = t1_img_sitk.GetDirection()
        dp['origin'] = t1_img_sitk.GetOrigin()
        pickle.dump(dp, f)

    t1km_sub = t1c_img - t1_img
    tmp = (t1c_img != 0) & (t1_img != 0)
    tmp = binary_fill_holes(tmp.astype(int))
    t1km_sub[~tmp.astype(bool)] = 0

    t1_img, bbox = extract_brain_region(t1_img, brain_mask, 0)
    t1c_img, bbox = extract_brain_region(t1c_img, brain_mask, 0)
    t2_img, bbox = extract_brain_region(t2_img, brain_mask, 0)
    flair_img, bbox = extract_brain_region(flair_img, brain_mask, 0)
    seg_img, bbox = extract_brain_region(seg_img, brain_mask, 0)

    assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape

    # Z- score: based on whole 3D image
    msk = t1_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1_img, msk, 2., 98.)
    t1_img[msk] = (t1_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t1c_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1c_img, msk, 2., 98.)
    t1c_img[msk] = (t1c_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t2_img != 0
    tmp = cut_off_values_upper_lower_percentile(t2_img, msk, 2., 98.)
    t2_img[msk] = (t2_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = flair_img != 0
    tmp = cut_off_values_upper_lower_percentile(flair_img, msk, 2., 98.)
    flair_img[msk] = (flair_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    shp = t1_img.shape
    # pad_size = np.max(np.vstack((np.array([128, 128, 128]), np.array(shp))), 0)
    pad_size = np.max(np.vstack((np.array([160, 192, 160]), np.array(shp))), 0)
    t1_img = reshape_by_padding_upper_coords(t1_img, pad_size, 0)
    t1c_img = reshape_by_padding_upper_coords(t1c_img, pad_size, 0)
    t2_img = reshape_by_padding_upper_coords(t2_img, pad_size, 0)
    flair_img = reshape_by_padding_upper_coords(flair_img, pad_size, 0)
    seg_img = reshape_by_padding_upper_coords(seg_img, pad_size, 0)

    all_data = np.zeros([5] + list(t1_img.shape), dtype=np.float32)
    all_data[0] = t1_img
    all_data[1] = t1c_img
    all_data[2] = t2_img
    all_data[3] = flair_img
    all_data[4] = seg_img

    np.save(os.path.join(out_folder, "%03.0d" % id), all_data)


# Data shape [d,w,h]
def simple_pre(t1_img, t1c_img, t2_img, flair_img, seg_img):
    brain_mask = (t1_img != t1_img[0, 0, 0]) & (t1c_img != t1c_img[0, 0, 0]) & (t2_img != t2_img[0, 0, 0]) & (
            flair_img != flair_img[0, 0, 0])

    t1_img, bbox = extract_brain_region(t1_img, brain_mask, 0)
    t1c_img, bbox = extract_brain_region(t1c_img, brain_mask, 0)
    t2_img, bbox = extract_brain_region(t2_img, brain_mask, 0)
    flair_img, bbox = extract_brain_region(flair_img, brain_mask, 0)
    seg_img, bbox = extract_brain_region(seg_img, brain_mask, 0)

    assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape

    shp = t1_img.shape
    # pad_size = np.max(np.vstack((np.array([160, 192, 160]), np.array(shp))), 0)
    pad_size = np.array([160, 192, 160])
    t1_img = reshape_by_padding_upper_coords(t1_img, pad_size, 0)
    t1c_img = reshape_by_padding_upper_coords(t1c_img, pad_size, 0)
    t2_img = reshape_by_padding_upper_coords(t2_img, pad_size, 0)
    flair_img = reshape_by_padding_upper_coords(flair_img, pad_size, 0)
    seg_img = reshape_by_padding_upper_coords(seg_img, pad_size, 0)

    all_data = np.zeros([5] + list(t1_img.shape), dtype=np.float32)
    all_data[0] = t1_img
    all_data[1] = t1c_img
    all_data[2] = t2_img
    all_data[3] = flair_img
    all_data[4] = seg_img

    return all_data
