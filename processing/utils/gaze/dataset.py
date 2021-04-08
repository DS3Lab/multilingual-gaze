import os

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

from processing.settings import LOGGER
from processing.utils.dataset import Dataset


class GazeDataset(Dataset):
    def __init__(self, cf, tokenizer, dir, task):
        super().__init__(tokenizer, dir, task)

        self.feature_max = cf.feature_max  # gaze features will be standardized between 0 and self.feature_max

    def read_pipeline(self):
        self.load_data()

        self.d_out = len(self.targets["train"][0][0])  # number of gaze features
        self.target_pad = -1

        self.standardize()
        self.tokenize_from_words()
        self.pad_targets()
        self.calc_input_ids()
        self.calc_attn_masks()
        self.calc_numpy()

    def load_data(self):
        LOGGER.info(f"Loading data for task {self.task}")
        for mode in self.modes:
            print(mode)
            dataset_pd = pd.read_csv(os.path.join(self.dir, mode + "_dataset.csv"),
                                     na_filter=False, index_col=0)
            word_func = lambda s: [w for w in s["word"].values.tolist()]
            features_func = lambda s: [np.array(s.drop(columns=["sentence_num", "word"]).iloc[i])
                                       for i in range(len(s))]

            self.text_inputs[mode] = dataset_pd.groupby("sentence_num").apply(word_func).tolist()
            print(len(self.text_inputs[mode]))
            self.targets[mode] = dataset_pd.groupby("sentence_num").apply(features_func).tolist()

        # check for duplicate sentence in train and test set
        dups = []
        for i, s in enumerate(self.text_inputs["train"]):
            if s in self.text_inputs["test"]:
                print("WARNING! Duplicate in test set....")
                dups.append(i)

        # remove duplicated from training data
        print(len(self.text_inputs["train"]))
        print(len(dups))
        for d in sorted(dups, reverse=True):
            del self.text_inputs["train"][d]
            del self.targets["train"][d]
        print(len(self.text_inputs["train"]))


    def standardize(self):
        """
        Standardizes the features between 0 and self.feature_max.
        """
        LOGGER.info(f"Standardizing target data for task {self.task}")
        features = self.targets["train"]
        scaler = MinMaxScaler(feature_range=[0, self.feature_max])
        flat_features = [j for i in features for j in i]
        scaler.fit(flat_features)

        self.targets["train"] = [list(scaler.transform(i)) for i in features]
        self.targets["val"] = [list(scaler.transform(i)) for i in self.targets["val"]]
        self.targets["test"] = [list(scaler.transform(i)) for i in self.targets["test"]]

        filen = os.path.join("scaled-test-"+self.task+".csv")

        print(filen)

        flat_preds = [j for i in self.targets["test"] for j in i]

        preds_pd = pd.DataFrame(flat_preds, columns=["n_fix", "first_fix_dur", "first_pass_dur",
                                                     "total_fix_dur", "mean_fix_dur", "fix_prob",
                                                     "n_refix", "reread_prob"])
        preds_pd.to_csv(filen)

        print("saved.")



    def pad_targets(self):
        """
        Adds the pad tokens in the positions of the [CLS] and [SEP] tokens, adds the pad
        tokens in the positions of the subtokens, and pads the targets with the pad token.
        """
        LOGGER.info(f"Padding targets for task {self.task}")
        for mode in self.modes:
            targets = [np.full((len(i), self.d_out), self.target_pad) for i in self.text_inputs[mode]]
            for k, (i, j) in enumerate(zip(self.targets[mode], self.maps[mode])):
                targets[k][j, :] = i

            target_pad_vector = np.full((1, self.d_out), self.target_pad)
            targets = [np.concatenate((target_pad_vector, i, target_pad_vector)) for i in targets]

            self.targets[mode] = pad_sequences(targets, value=self.target_pad, padding="post")

    def calc_numpy(self):
        LOGGER.info(f"Calculating numpy arrays for task {self.task}")
        for mode in self.modes:
            input_numpy = np.asarray(self.text_inputs[mode], dtype=np.int64)
            mask_numpy = np.asarray(self.masks[mode], dtype=np.float32)
            target_numpy = np.asarray(self.targets[mode], dtype=np.float32)

            self.numpy[mode] = list(zip(input_numpy, target_numpy, mask_numpy))
