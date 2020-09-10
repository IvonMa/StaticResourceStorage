import numpy as np
import SimpleITK as sitk
import os
import config as paths


def convert_to_original_coord_system(seg_pred, pat_in_dataset):
    orig_shape = pat_in_dataset['orig_shp']
    # axis order is z x y
    brain_bbox_z = pat_in_dataset['bbox_z']
    brain_bbox_x = pat_in_dataset['bbox_x']
    brain_bbox_y = pat_in_dataset['bbox_y']
    new_seg = np.zeros(orig_shape, dtype=np.uint8)
    tmp_z = np.min((orig_shape[0], brain_bbox_z[0] + seg_pred.shape[0]))
    tmp_x = np.min((orig_shape[1], brain_bbox_x[0] + seg_pred.shape[1]))
    tmp_y = np.min((orig_shape[2], brain_bbox_y[0] + seg_pred.shape[2]))
    new_seg[brain_bbox_z[0]:tmp_z, brain_bbox_x[0]:tmp_x, brain_bbox_y[0]:tmp_y] = seg_pred[:tmp_z - brain_bbox_z[0],
                                                                                   :tmp_x - brain_bbox_x[0],
                                                                                   :tmp_y - brain_bbox_y[0]]
    return new_seg


def save_val_dataset_as_nifti(results_dir=os.path.join(paths.results_folder, "final"),
                              out_dir=os.path.join(paths.results_folder, "val_set_results_new")):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    a = load_dataset(folder=paths.preprocessed_validation_data_folder)
    for pat in a.keys():

        b = convert_to_original_coord_system(a[pat]['data'], a[pat])
        sitk_img = sitk.GetImageFromArray(b)
        sitk_img.SetSpacing(a[pat]['spacing'])
        sitk_img.SetDirection(a[pat]['direction'])
        sitk_img.SetOrigin(a[pat]['origin'])
        sitk.WriteImage(sitk_img, os.path.join(out_dir, a[pat]['name'] + ".nii.gz"))


import pickle


def load_dataset(pat_ids=range(300), folder=paths.preprocessed_training_data_folder):
    id_name_conversion = np.loadtxt(os.path.join(folder, "id_name_conversion.txt"), dtype="str")
    idxs = id_name_conversion[:, 1].astype(int)
    dataset = {}
    for p in pat_ids:
        if os.path.isfile(os.path.join(folder, "%03.0d.npy" % p)):
            dataset[p] = {}
            dataset[p]['data'] = np.load(os.path.join(folder, "%03.0d.npy" % p), mmap_mode='r')
            dataset[p]['idx'] = p
            dataset[p]['name'] = id_name_conversion[np.where(idxs == p)[0][0], 0]
            dataset[p]['type'] = id_name_conversion[np.where(idxs == p)[0][0], 2]

            with open(os.path.join(folder, "%03.0d.pkl" % p), 'r') as f:
                dp = pickle.load(f)

            dataset[p]['orig_shp'] = dp['orig_shp']
            dataset[p]['bbox_z'] = dp['bbox_z']
            dataset[p]['bbox_x'] = dp['bbox_x']
            dataset[p]['bbox_y'] = dp['bbox_y']
            dataset[p]['spacing'] = dp['spacing']
            dataset[p]['direction'] = dp['direction']
            dataset[p]['origin'] = dp['origin']
    return dataset


def save_pred_mask_nii(out_dir, pred_mask, patient_id, folder=paths.preprocessed_testing_data_folder):
    # Getting its original size data
    with open(os.path.join(folder, "%03.0d.pkl" % patient_id), 'rb') as f:
        dp = pickle.load(f)
    # Converting it to original size
    b = convert_to_original_coord_system(pred_mask, dp)

    # Get its id-name conversion
    id_name_conversion = np.loadtxt(os.path.join(folder, "id_name_conversion.txt"), dtype="str")
    idxs = id_name_conversion[:, 1].astype(int)
    patient_name = id_name_conversion[np.where(idxs == patient_id)[0][0], 0]
    patient_type = id_name_conversion[np.where(idxs == patient_id)[0][0], 2]
    # Save it
    sitk_img = sitk.GetImageFromArray(b)
    sitk_img.SetSpacing(dp['spacing'])
    sitk_img.SetDirection(dp['direction'])
    sitk_img.SetOrigin(dp['origin'])
    sitk.WriteImage(sitk_img, os.path.join(out_dir, patient_name + ".nii.gz"))



if __name__ == '__main__':
    # Test:  save_pred_mask_nii method
    # Reshape and save a labelled mask. So if the dice coefficient is 1 then it works well.
    original_file_path = '/Users/mayunfeng/PycharmProjects/data/Dataset_test/LGG/BraTS19_TCIA12_470_1/BraTS19_TCIA12_470_1_seg.nii.gz'
    labelled_mask_original = sitk.GetArrayFromImage(sitk.ReadImage(original_file_path)).astype(np.float32)
    patient_id = 46
    folder = paths.preprocessed_testing_data_folder
    preprocessed_file_path = os.path.join(folder, "%03.0d.npy" % patient_id)
    pre_processed_file = np.load(preprocessed_file_path)[4]
    save_pred_mask_nii(paths.segmentation_result_folder, pre_processed_file, patient_id)

    labelled_mask_from_reshape_path = os.path.join(paths.segmentation_result_folder, 'BraTS19_TCIA12_470_1.nii.gz')
    labelled_mask_from_reshape = sitk.GetArrayFromImage(sitk.ReadImage(labelled_mask_from_reshape_path)).astype(np.float32)

    import losses_wolny
    import torch

    labelled_mask_from_reshape_tensor = torch.from_numpy(labelled_mask_from_reshape).unsqueeze_(dim=0).unsqueeze_(dim=0)
    labelled_mask_original_tensor = torch.from_numpy(labelled_mask_original).unsqueeze_(dim=0).unsqueeze_(dim=0)
    DSE = losses_wolny.compute_per_channel_dice(labelled_mask_from_reshape_tensor, labelled_mask_original_tensor)
    print(DSE)
