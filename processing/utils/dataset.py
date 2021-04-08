import random
from abc import ABC

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences

from processing.settings import LOGGER


class Dataset(ABC):
    def __init__(self, tokenizer, dir, task):
        self.tokenizer = tokenizer  # tokenizer for the BERT model
        self.dir = dir
        self.task = task

        self.modes = ["train", "val", "test"]

        self.text_inputs = {}
        self.targets = {}
        self.masks = {}  # split key padding attention masks for the BERT model
        self.maps = {}  # split mappings between tokens and original words
        self.numpy = {}  # split numpy arrays, ready for the model

    def tokenize_and_map(self, sentence):
        """
        Tokenizes a sentence, and returns the tokens and a list of starting indices of the original words.
        """
        tokens = []
        map = []

        for w in sentence:
            map.append(len(tokens))
            tokens.extend(self.tokenizer.tokenize(w) if self.tokenizer.tokenize(w) else [self.tokenizer.unk_token])

        return tokens, map

    def tokenize_from_words(self):
        """
        Tokenizes the sentences in the dataset with the pre-trained tokenizer, storing the start index of each word.
        """
        LOGGER.info(f"Tokenizing sentences for task {self.task}")
        for mode in self.modes:
            print(mode)
            print(len(self.text_inputs[mode]))
            tokenized = []
            maps = []

            for s in self.text_inputs[mode]:
                tokens, map = self.tokenize_and_map(s)

                tokenized.append(tokens)
                maps.append(map)
                #print(tokens)
            print("max tokenized seq len: ", max(len(l) for l in tokenized))

            self.text_inputs[mode] = tokenized
            self.maps[mode] = maps

    def calc_input_ids(self):
        """
        Converts tokens to ids for the BERT model.
        """
        LOGGER.info(f"Calculating input ids for task {self.task}")
        for mode in self.modes:
            ids = [self.tokenizer.prepare_for_model(self.tokenizer.convert_tokens_to_ids(s))["input_ids"]
                   for s in self.text_inputs[mode]]
            self.text_inputs[mode] = pad_sequences(ids, value=self.tokenizer.pad_token_id, padding="post")

    def calc_attn_masks(self):
        """
        Calculates key paddding attention masks for the BERT model.
        """
        LOGGER.info(f"Calculating attention masks for task {self.task}")
        for mode in self.modes:
            self.masks[mode] = [[j != self.tokenizer.pad_token_id for j in i] for i in self.text_inputs[mode]]


class TaskDataset(Dataset):
    def __init__(self, cf, tokenizer, device, baseline, dir, task, predict_percs=None,
                 gaze_tokenizer=None, gaze_models=None, gaze_max=None, d_gaze=None):
        super().__init__(tokenizer, dir, task)

        self.baseline = baseline  # boolean flag, whether to predict gaze or not
        self.predict_percs = predict_percs
        self.gaze_tokenizer = gaze_tokenizer  # tokenizer for the gaze models
        self.gaze_models = gaze_models
        self.gaze_max = gaze_max  # maximum value of the gaze features, for quantization
        self.d_gaze = d_gaze  # number of gaze features
        self.gaze_bs = cf.gaze_bs
        self.n_bins = cf.n_bins  # number of bins for quantizing the gaze features
        self.device = device

        self.gaze_pad_idx = cf.n_bins
        self.gaze_start_idx = cf.n_bins + 1
        self.gaze_end_idx = cf.n_bins + 2
        self.gaze_vocab_size = cf.n_bins + 3

        self.gaze_inputs = {}

    def ensemble_predict(self, batch_idxs, mode):
        """
        Predicts gaze features using the ensemble of gaze models.
        """
        b_sentences = [self.text_inputs[mode][i] for i in batch_idxs]
        b_tokens, b_maps = list(map(list, zip(*[self.tokenize_and_map(s) for s in b_sentences])))
        b_ids = [self.gaze_tokenizer.prepare_for_model(self.gaze_tokenizer.convert_tokens_to_ids(s))["input_ids"]
                 for s in b_tokens]
        b_inputs = pad_sequences(b_ids, value=self.gaze_tokenizer.pad_token_id, padding="post")
        b_masks = [[j != self.gaze_tokenizer.pad_token_id for j in i] for i in b_inputs]
        outputs = []

        with torch.no_grad():
            b_inputs = torch.tensor(np.array(b_inputs), dtype=torch.int64).to(self.device)
            b_masks = torch.tensor(np.array(b_masks), dtype=torch.float32).to(self.device)

            for model in self.gaze_models:
                model.to(self.device)
                model.eval()

                b_outputs = model(b_inputs, attention_mask=b_masks)[0]

                b_outputs_orig_len = []
                for o, m in zip(b_outputs, b_maps):
                    o = o[1:, :]
                    b_outputs_orig_len.append([o[i] for i in m])

                outputs.append([[j.cpu().numpy() for j in i] for i in b_outputs_orig_len])

        transpose_outputs = list(map(list, zip(*outputs)))
        sum_outputs = [sum([np.asarray(j) for j in i]) for i in transpose_outputs]
        avg_outputs = [i / len(self.gaze_models) for i in sum_outputs]

        for k, i in enumerate(batch_idxs):
            self.gaze_inputs[mode][i] = avg_outputs[k]

    def predict_gaze(self):
        """
        Predicts or uniformly samples gaze features.
        """
        LOGGER.info(f"Predicting gaze data for task {self.task}")
        for mode in self.modes:
            self.gaze_inputs[mode] = [np.full((len(i), self.d_gaze), np.nan) for i in self.text_inputs[mode]]

            len_predict = round(len(self.text_inputs[mode]) * self.predict_percs[mode])
            predict_idxs = random.sample(range(len(self.text_inputs[mode])), k=len_predict)

            for batch_index in range(0, len(predict_idxs), self.gaze_bs):
                batch_idxs = predict_idxs[batch_index:batch_index + self.gaze_bs]
                self.ensemble_predict(batch_idxs, mode)

    def digitize_gaze(self):
        """
        Quantizes gaze features in self.n_bins bins.
        """
        LOGGER.info(f"Digitizing gaze data for task {self.task}")
        bins = np.linspace(0, self.gaze_max, self.n_bins - 1)

        for mode in self.modes:
            self.gaze_inputs[mode] = [np.digitize(i, bins) if not np.isnan(i).any() else i for i in
                                      self.gaze_inputs[mode]]

    def tokenize_preserve_and_map(self, sentence, gaze):
        """
        Tokenizes a list of words with the pretrained tokenizer, and assigns
        to the sub-word tokens the same gaze features as the parent word.
        """
        tokenized = []
        gazes = []
        map = []

        for w, g in zip(sentence, gaze):
            map.append(len(tokenized))
            tokens = self.tokenizer.tokenize(w) if self.tokenizer.tokenize(w) else [self.tokenizer.unk_token]
            tokenized.extend(tokens)

            n_subwords = len(tokens)
            gazes.extend([g] * n_subwords)

        return tokenized, gazes, map

    def tokenize_with_gaze(self):
        """
        Tokenizes the sentences in the dataset with the pre-trained
        tokenizer, preserving targets and predicted gaze features order.
        """
        LOGGER.info(f"Tokenizing sentences for task {self.task}")
        for mode in self.modes:
            tokenized = []
            gazes = []
            maps = []

            for s, g in zip(self.text_inputs[mode], self.gaze_inputs[mode]):
                tokens, gaze, map = self.tokenize_preserve_and_map(s, g)

                tokenized.append(tokens)
                gazes.append(gaze)
                maps.append(map)

            self.text_inputs[mode] = tokenized
            self.gaze_inputs[mode] = gazes
            self.maps[mode] = maps

    def pad_gaze(self):
        """
        Adds the pad tokens in the positions of the [CLS] and [SEP]
        tokens, and pads the gaze sequences with the pad token.
        """
        LOGGER.info(f"Padding gaze data for task {self.task}")
        for mode in self.modes:
            gaze_pad_idxs = np.full((1, self.d_gaze), self.gaze_pad_idx)
            gaze_start_idxs = np.full((1, self.d_gaze), self.gaze_start_idx)
            gaze_end_idxs = np.full((1, self.d_gaze), self.gaze_end_idx)
            gaze_inputs = [np.concatenate((gaze_start_idxs, i, gaze_end_idxs)) for i in self.gaze_inputs[mode]]

            self.gaze_inputs[mode] = pad_sequences(gaze_inputs, value=gaze_pad_idxs, padding="post")

    def dummy_gaze(self):
        """
        Builds dummy gaze numpy arrays filled with nans.
        """
        LOGGER.info(f"Assigning dummy gaze for task {self.task}")
        for mode in self.modes:
            self.gaze_inputs[mode] = np.full((len(self.text_inputs[mode]), 2, 1), np.nan)
