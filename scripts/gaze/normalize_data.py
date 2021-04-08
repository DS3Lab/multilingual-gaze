from settings import reproducibility_setup

reproducibility_setup()

import argparse
import json
import os

from processing import GazeDataNormalizer, LOGGER


def main(args):
    data_gaze_dir = args.data_gaze_dir
    tasks = args.tasks
    split_percs = args.split_percs

    datanormalizers = [GazeDataNormalizer(os.path.join(data_gaze_dir, task), task, split_percs) for task in tasks]

    for dn in datanormalizers:
        dn.read_pipeline()
        dn.save_datasets()

    LOGGER.info("Datasets saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dg", "--data_gaze_dir", type=str,
                        default="/Users/fedepiro/Git/ceiling-eye-nlp/data/gaze")
    parser.add_argument("-ts", "--tasks", type=str, nargs="+",
                        default=["dundee", "geco", "zuco11", "zuco12", "zuco21"])
    parser.add_argument("-sp", "--split_percs", type=json.loads,
                        default='{"train": 0.9, "val": 0.05, "test": 0.05}')
    args = parser.parse_args()
    main(args)
