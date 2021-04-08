from settings import reproducibility_setup

reproducibility_setup()

import argparse
import os

import pandas as pd

from processing import LOGGER


def main(args):
    data_gaze_dir = args.data_gaze_dir
    tasks = args.tasks
    data_precentage = args.percentage

    all_gaze_dir = os.path.join(data_gaze_dir, "all-"+data_precentage)
    if not os.path.exists(all_gaze_dir):
        os.makedirs(all_gaze_dir)

    LOGGER.info("Combining gaze datasets")
    modes = ["train", "val", "test"]

    for mode in modes:
        first = True
        for task in tasks:
            #LOGGER.info(mode, task)
            print(mode, task)
            if mode != "train":
                data_precentage = 1.0
            if first:
                print(os.path.join(data_gaze_dir, task, mode + "_dataset.csv"))
                dataset_pd = pd.read_csv(os.path.join(data_gaze_dir, task, mode + "_dataset.csv"),
                                         na_filter=False, index_col=0)
                first = False
                x = round(len(dataset_pd) * float(data_precentage))
                print(len(dataset_pd))
                dataset_pd = dataset_pd[:x]
                print(len(dataset_pd))

            else:
                print(os.path.join(data_gaze_dir, task, mode + "_dataset.csv"))
                to_append = pd.read_csv(os.path.join(data_gaze_dir, task, mode + "_dataset.csv"),
                                                           na_filter=False, index_col=0)
                x = round(len(to_append) * float(data_precentage))
                print(len(to_append))
                to_append = to_append[:x]
                print(len(to_append))

                # re-index sentence numbers to avoid conflicts between multiple datasets
                i = 1
                for index, row in to_append.iterrows():
                    if row["sentence_num"] in dataset_pd["sentence_num"]:
                        if index != len(to_append) - 1:
                            if to_append.at[index + 1, 'sentence_num'] == row["sentence_num"]:
                                to_append.at[index, 'sentence_num'] = dataset_pd["sentence_num"].max()+i
                            else:
                                to_append.at[index, 'sentence_num'] = dataset_pd["sentence_num"].max() + i
                                i+=1
                        else:
                            to_append.at[index, 'sentence_num'] = dataset_pd["sentence_num"].max() + i

                dataset_pd = dataset_pd.append(to_append)

        print("Number of sentences:", dataset_pd["sentence_num"].nunique())
        print("Number of tokens:", len(dataset_pd))
        dataset_pd = dataset_pd.reset_index(drop=True)
        dataset_pd.to_csv(os.path.join(all_gaze_dir, mode + "_dataset.csv"))
        LOGGER.info("Dataset saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dg", "--data_gaze_dir", type=str,
                        default="/Users/fedepiro/Git/ceiling-eye-nlp/data/gaze")
    parser.add_argument("-ts", "--tasks", type=str, nargs="+",
                        default=["dundee", "geco", "zuco11", "zuco12", "zuco21"])
    parser.add_argument("-pe", "--percentage", type=str,
                        default="1")
    args = parser.parse_args()
    main(args)
