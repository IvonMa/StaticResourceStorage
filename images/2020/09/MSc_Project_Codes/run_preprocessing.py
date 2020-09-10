"""
    Codes refer to the implementation of Isensee et al. https://github.com/MIC-DKFZ/BraTS2017
    Modified by MA Yunfeng
"""

import argparse

import config
from pre_processing import run_preprocessing_BraTS2017_trainSet, run_preprocessing_BraTS2017_valOrTestSet

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be train or testval. Use testval for validation and test datasets",
                    type=str)
args = parser.parse_args()

if args.mode == "val":
    run_preprocessing_BraTS2017_valOrTestSet(config.raw_validation_data_folder, config.preprocessed_validation_data_folder)
elif args.mode == "train":
    run_preprocessing_BraTS2017_trainSet(config.raw_training_data_folder, config.preprocessed_training_data_folder)
elif args.mode == "test":
    run_preprocessing_BraTS2017_trainSet(config.raw_testing_data_folder, config.preprocessed_testing_data_folder)
else:
    raise ValueError("Unknown value for --mode. Use \"train\", \"test\" or \"val\"")
