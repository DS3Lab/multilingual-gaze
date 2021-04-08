import os
import random
import h5py
import numpy as np
import pandas as pd
import scipy.io
from itertools import islice

from processing.settings import LOGGER


def get_mean_fix_dur(n_fix, total_fix_dur):
    return 0 if n_fix == 0 else total_fix_dur / n_fix

def get_fix_prob(n_fix):
    return int(n_fix > 0)


def get_n_refix(n_fix):
    return max([n_fix - 1, 0])


def get_reread_prob(n_refix):
    return int(n_refix > 0)


class DundeeDataNormalizer:
    def __init__(self, dir, task):
        self.dir = dir
        self.task = task

        self.files = {}  # dict indexed by frag and subj, each entry is a 2-item list containing the relevant fpaths
        self.frags = []
        self.subjs = []
        self.words_idxs = {}  # dict indexed by frag and subj, each entry is a 3-tuple storing word and indexes of
        # first appearance in files 1 and 2
        self.frags_words = {}  # dict indexed by frag, each entry is the list of words
        self.frags_features = {}  # dict indexed by frag, each entry is the list of numpy arrays containing features
        self.flat_words = []
        self.flat_features = []

    def read_files(self):
        LOGGER.info(f"Reading files for task {self.task}")
        for file in sorted(os.listdir(self.dir)):
            if not file.endswith(".dat"):
                continue
            frag = int(file[2:4])
            subj = file[:2]
            fpath = os.path.join(self.dir, file)
            self.add_file(fpath, frag, subj)

    def add_file(self, fpath, frag, subj):
        if frag not in self.files:
            self.files[frag] = {subj: [fpath]}
        elif subj not in self.files[frag]:
            self.files[frag][subj] = [fpath]
        else:
            self.files[frag][subj].append(fpath)

    def read_frags(self):
        self.frags = [frag for frag in self.files]

    def read_subjs(self):
        self.subjs = [subj for subj in self.files[self.frags[0]]]
        print(self.subjs)

    @staticmethod
    def get_n_fix(w, lines_2):
        cnt = 0
        for l in lines_2:
            if l[0] != w:
                break
            elif l[1] != -99:
                cnt += 1
        return cnt

    @staticmethod
    def get_first_fix_dur(lines_1):
        return lines_1[0][7]

    @staticmethod
    def get_first_pass_dur(w, lines_1):
        tot = 0
        for l in lines_1:
            if l[0] != w:
                break
            else:
                tot += l[7]
        return tot

    @staticmethod
    def get_total_fix_dur(w, lines_2):
        tot = 0
        for l in lines_2:
            if l[0] != w:
                break
            else:
                tot += l[7]
        return tot

    @staticmethod
    def get_first_occurrence(word, lines):
        """
        Returns index of first occurrence of "word" in "lines".
        """
        for i, l in enumerate(lines):
            w = l.split()[0]
            if w == word:
                return i

    def read_words(self):
        """
        Calculates "self.words_idxs" and "self.frags_words".
        """
        words_idxs_tmp = {}
        for frag in self.frags:
            self.frags_words[frag] = []
            self.words_idxs[frag] = {}
            words_idxs_tmp[frag] = {}
            for i, subj in enumerate(self.subjs):
                self.words_idxs[frag][subj] = {}
                words_idxs_tmp[frag][subj] = []
                with open(self.files[frag][subj][0], errors="ignore") as f:
                    lines_1 = f.readlines()[1:]
                with open(self.files[frag][subj][1], errors="ignore") as f:
                    lines_2 = f.readlines()[1:]

                last_occurrences_1 = {}
                for j, l in enumerate(lines_2):
                    w = l.split()[0]
                    fix = int(l.split()[1])
                    if fix == -99:
                        words_idxs_tmp[frag][subj].append((w, -1, j))
                        continue
                    if w not in last_occurrences_1:
                        last = -1
                    else:
                        last = last_occurrences_1[w]
                    idx_1 = last + 1 + self.get_first_occurrence(w, lines_1[last + 1:])
                    last_occurrences_1[w] = idx_1
                    words_idxs_tmp[frag][subj].append((w, idx_1, j))  # words_idxs_tmp stores for each word the two
                    # indexes for files 1 and 2 corresponding to the same data entry

                # reduces words_idxs_tmp to just the first occurrence of each word
                last_w = None
                j = 0
                for k in words_idxs_tmp[frag][subj]:
                    w = k[0]
                    if last_w is None or w != last_w:
                        if i == 0:
                            self.frags_words[frag].append(w)
                        self.words_idxs[frag][subj][j] = k
                        last_w = w
                        j += 1

    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")
        for i, frag in enumerate(self.frags):
            LOGGER.info(f"Processing fragment {i + 1} out of {len(self.frags)}")

            empty = [[None] * len(self.subjs)] * len(self.frags_words[frag])
            nfx_pd = pd.DataFrame(empty, columns=self.subjs)
            ffd_pd = pd.DataFrame(empty, columns=self.subjs)
            fpd_pd = pd.DataFrame(empty, columns=self.subjs)
            tfd_pd = pd.DataFrame(empty, columns=self.subjs)
            mfd_pd = pd.DataFrame(empty, columns=self.subjs)
            fxp_pd = pd.DataFrame(empty, columns=self.subjs)
            nrfx_pd = pd.DataFrame(empty, columns=self.subjs)
            rrdp_pd = pd.DataFrame(empty, columns=self.subjs)


            for j, subj in enumerate(self.subjs):
                with open(self.files[frag][subj][0], errors="ignore") as f:
                    lines = [l.split() for l in f.readlines()[1:]]
                    lines_1 = [[l[0]] + [int(i) for i in l[1:]] for l in lines]
                with open(self.files[frag][subj][1], errors="ignore") as f:
                    lines = [l.split() for l in f.readlines()[1:]]
                    lines_2 = [[l[0]] + [int(i) for i in l[1:]] for l in lines]
                for k, w in enumerate(self.frags_words[frag]):
                    idx_1 = self.words_idxs[frag][subj][k][1]
                    idx_2 = self.words_idxs[frag][subj][k][2]

                    nfx = self.get_n_fix(w, lines_2[idx_2:])
                    ffd = 0 if idx_1 == -1 else self.get_first_fix_dur(lines_1[idx_1:])
                    fpd = 0 if idx_1 == -1 else self.get_first_pass_dur(w, lines_1[idx_1:])
                    tfd = self.get_total_fix_dur(w, lines_2[idx_2:])

                    # print(nfx_ls[k][j], tfd_ls[k][j])
                    mfd = get_mean_fix_dur(nfx, tfd)
                    fxp = get_fix_prob(nfx)
                    nrfx = get_n_refix(nfx)
                    rrdp = get_reread_prob(nrfx)

                    nfx_pd[subj][k] = nfx
                    ffd_pd[subj][k] = ffd
                    fpd_pd[subj][k] = fpd
                    tfd_pd[subj][k] = tfd
                    mfd_pd[subj][k] = mfd
                    fxp_pd[subj][k] = fxp
                    nrfx_pd[subj][k] = nrfx
                    rrdp_pd[subj][k] = rrdp

            nfx = nfx_pd.mean(axis=1).tolist()
            ffd = ffd_pd.mean(axis=1).tolist()
            fpd = fpd_pd.mean(axis=1).tolist()
            tfd = tfd_pd.mean(axis=1).tolist()
            mfd = mfd_pd.mean(axis=1).tolist()
            fxp = fxp_pd.mean(axis=1).tolist()
            nrfx = nrfx_pd.mean(axis=1).tolist()
            rrdp = rrdp_pd.mean(axis=1).tolist()

            features = [nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]
            self.frags_features[frag] = [np.array(i) for i in list(zip(*features))]
            self.flat_words.extend([w for w in self.frags_words[frag]])
            self.flat_features.extend([f for f in self.frags_features[frag]])


class GECODataNormalizer:
    def __init__(self, dir, task, print_every=3000):
        self.dir = dir
        self.task = task
        self.print_every = print_every

        if self.task == "geco-nl":
            self.full_subj = ["pp03", "pp04", "pp16"]  # subjects with complete data
        else:
            self.full_subj = "pp21"

        self.frags = []
        self.subjs = []
        self.flat_words = []

    def read_file(self):
        LOGGER.info(f"Reading file for task {self.task}")
        if self.task == "geco-nl":
            # Dutch part of the GECO corpus
            self.file = pd.read_excel(os.path.join(self.dir, "L1ReadingData.xlsx"), na_filter=False)
        else:
            # English part of the GECO corpus
            self.file = pd.read_excel(os.path.join(self.dir, "MonolingualReadingData.xlsx"), na_filter=False)

    def read_frags(self):
        self.frags = sorted(self.file["PART"].unique())

    def read_subjs(self):
        """
        Reads list of subjs and sorts it such that the full subject is in the first position.
        """
        LOGGER.info(f"Reading subjects for task {self.task}")
        self.subjs = sorted(self.file["PP_NR"].unique())
        for i, subj in enumerate(self.subjs):
            if subj == self.full_subj:
                break
        self.subjs.insert(0, self.subjs.pop(i))

    def read_words(self):
        """
        Word list is extracted from the full subject.
        """


        LOGGER.info(f"Reading words for task {self.task}")
        if self.task == "geco-nl":
            for frag in self.frags:
                for s in self.full_subj:
                    flt_file = self.file
                    isfrag = flt_file["PART"] == frag
                    issubj = flt_file["PP_NR"] == s
                    flt_file = flt_file[isfrag]
                    flt_file = flt_file[issubj]
                    self.flat_words.extend([str(w) for w in flt_file["WORD"].tolist()])
            print(len(self.flat_words))



        else:
            for frag in self.frags:
                flt_file = self.file
                isfrag = flt_file["PART"] == frag
                issubj = flt_file["PP_NR"] == self.full_subj
                flt_file = flt_file[isfrag]
                flt_file = flt_file[issubj]
                self.flat_words.extend([str(w) for w in flt_file["WORD"].tolist()])
            print(len(self.flat_words))

    def match(self, pattern):
        """
        Finds all occurrences of "pattern" (list of words) in "self.flat_words".
        """
        return [i for i in range(len(self.flat_words) - len(pattern) + 1)
                if self.flat_words[i:i + len(pattern)] == pattern]

    def find_idx(self, idx):
        """
        Finds index of the word at dataset index "i" in the flat list of words "self.flat_words". This is
        performed by searching the word and its surrounding words in the flat list of words, taking
        care of possible missings before and after the word. Width of context is increased until we find
        a unique match.
        """
        word = str(self.file.iloc[idx]["WORD"])
        matches = None
        beam = 0

        while matches is None or len(matches) > 1:
            beam += 1
            prev_start = idx - beam
            if prev_start < 0:
                prev_words = [str(i) for i in self.file.iloc[:idx]["WORD"].tolist()]
                offset = idx
            else:
                prev_words = [str(i) for i in self.file.iloc[prev_start:idx]["WORD"].tolist()]
                offset = beam
            nxt_end = idx + beam
            if nxt_end > len(self.file):
                nxt_words = [str(i) for i in self.file.iloc[idx:]["WORD"].tolist()]
            else:
                nxt_words = [str(i) for i in self.file.iloc[idx + 1:idx + 1 + beam]["WORD"].tolist()]

            pattern = prev_words + [word] + nxt_words
            matches = self.match(pattern)
            if len(matches) == 0:
                pattern = prev_words + [word]
                matches = self.match(pattern)
                if len(matches) == 0:
                    pattern = [word] + nxt_words
                    matches = self.match(pattern)
                    offset = 0

        return matches[0] + offset

    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")
        # for each feature, we build a pd.dataframe with shape (len(words), len(subjs)) to be filled with data; data
        # will be averaged over the subject dimension (columns)
        empty = [[None] * len(self.subjs)] * len(self.flat_words)
        nfx_pd = pd.DataFrame(empty, columns=self.subjs)
        ffd_pd = pd.DataFrame(empty, columns=self.subjs)
        fpd_pd = pd.DataFrame(empty, columns=self.subjs)
        tfd_pd = pd.DataFrame(empty, columns=self.subjs)
        mfd_pd = pd.DataFrame(empty, columns=self.subjs)
        fxp_pd = pd.DataFrame(empty, columns=self.subjs)
        nrfx_pd = pd.DataFrame(empty, columns=self.subjs)
        rrdp_pd = pd.DataFrame(empty, columns=self.subjs)

        for i in range(len(self.file)):
            if i % self.print_every == 0:
                LOGGER.info(f"Processing line {i + 1} out of {len(self.file)}")

            row = self.file.iloc[i]
            subj = row["PP_NR"]

            word_index = self.find_idx(i)  # i is the index iterating through all the lines of the dataset,
                # word_index is the index of the corresponding word in the flat words list; this calculation requires
                # some care because not all subjects are complete and the list of missings is not reported
            #print(row['WORD_ID'], row['WORD'])

            nfx = 0 if row["WORD_FIXATION_COUNT"] == "." else row["WORD_FIXATION_COUNT"]
            ffd = 0 if row["WORD_FIRST_FIXATION_DURATION"] == "." else row["WORD_FIRST_FIXATION_DURATION"]
            fpd = 0 if row["WORD_GAZE_DURATION"] == "." else row["WORD_GAZE_DURATION"]
            tfd = 0 if row["WORD_TOTAL_READING_TIME"] == "." else row["WORD_TOTAL_READING_TIME"]
            mfd = get_mean_fix_dur(nfx, tfd)
            fxp = get_fix_prob(nfx)
            nrfx = get_n_refix(nfx)
            rrdp = get_reread_prob(nrfx)

            nfx_pd[subj][word_index] = nfx
            ffd_pd[subj][word_index] = ffd
            fpd_pd[subj][word_index] = fpd
            tfd_pd[subj][word_index] = tfd
            mfd_pd[subj][word_index] = mfd
            fxp_pd[subj][word_index] = fxp
            nrfx_pd[subj][word_index] = nrfx
            rrdp_pd[subj][word_index] = rrdp

        nfx = nfx_pd.mean(axis=1).tolist()
        ffd = ffd_pd.mean(axis=1).tolist()
        fpd = fpd_pd.mean(axis=1).tolist()
        tfd = tfd_pd.mean(axis=1).tolist()
        mfd = mfd_pd.mean(axis=1).tolist()
        fxp = fxp_pd.mean(axis=1).tolist()
        nrfx = nrfx_pd.mean(axis=1).tolist()
        rrdp = rrdp_pd.mean(axis=1).tolist()

        features = [nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]
        self.flat_features = [np.array(i) for i in list(zip(*features))]

        print(len(self.flat_words))
        print(len(self.flat_features))


class ZuCo1DataNormalizer:
    def __init__(self, dir, task):
        self.dir = dir
        self.task = task

        self.full_subj = "ZAB"  # subject with complete data

        self.files = {}  # dict indexed by subj containing fpaths of corresponding .mat files
        self.subjs = []
        self.flat_words = []
        self.flat_features = []

    def read_files(self):
        LOGGER.info(f"Reading files for task {self.task}")
        for file in sorted(os.listdir(self.dir)):
            if not file.endswith(".mat"):
                continue
            subj = file.split("_")[0][-3:]
            fpath = os.path.join(self.dir, file)
            self.files[subj] = fpath

    def read_subjs(self):
        """
        Reads list of subjs and sorts it such that the full subject is in the first position.
        """
        self.subjs = [subj for subj in self.files]
        for i, subj in enumerate(self.subjs):
            if subj == self.full_subj:
                break
        self.subjs.insert(0, self.subjs.pop(i))

    def read_words(self):
        mat = scipy.io.loadmat(self.files[self.full_subj])
        sentence_data = mat["sentenceData"][0]
        for row in sentence_data:
            word_data = row["word"][0]
            for item in word_data:
                self.flat_words.append(item["content"][0])

    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")
        #  dicts indexed by sentence number and word number inside the sentence
        nfx_ls = {}
        ffd_ls = {}
        fpd_ls = {}
        tfd_ls = {}
        mfd_ls = {}
        fxp_ls = {}
        nrfx_ls = {}
        rrdp_ls = {}

        for i, subj in enumerate(self.subjs):
            LOGGER.info(f"Processing subject {i + 1} out of {len(self.subjs)}")
            mat = scipy.io.loadmat(self.files[subj])
            sentence_data = mat["sentenceData"][0]

            for j, row in enumerate(sentence_data):
                word_data = row["word"][0]
                cont = False
                try:
                    cont = np.isnan(word_data[0])
                except:
                    pass
                if cont:
                    continue

                if i == 0:
                    nfx_ls[j] = [[] for _ in range(len(word_data))]
                    ffd_ls[j] = [[] for _ in range(len(word_data))]
                    fpd_ls[j] = [[] for _ in range(len(word_data))]
                    tfd_ls[j] = [[] for _ in range(len(word_data))]
                    mfd_ls[j] = [[] for _ in range(len(word_data))]
                    fxp_ls[j] = [[] for _ in range(len(word_data))]
                    nrfx_ls[j] = [[] for _ in range(len(word_data))]
                    rrdp_ls[j] = [[] for _ in range(len(word_data))]

                for k, item in enumerate(word_data):
                    nfx_ls[j][k].append(0 if len(item["nFixations"]) == 0 else item["nFixations"][0][0])
                    ffd_ls[j][k].append(0 if len(item["FFD"]) == 0 else item["FFD"][0][0])
                    fpd_ls[j][k].append(0 if len(item["GD"]) == 0 else item["GD"][0][0])
                    tfd_ls[j][k].append(0 if len(item["TRT"]) == 0 else item["TRT"][0][0])
                    mfd_ls[j][k].append(get_mean_fix_dur(nfx_ls[j][k][-1], tfd_ls[j][k][-1]))
                    fxp_ls[j][k].append(get_fix_prob(nfx_ls[j][k][-1]))
                    nrfx_ls[j][k].append(get_n_refix(nfx_ls[j][k][-1]))
                    rrdp_ls[j][k].append(get_reread_prob(nrfx_ls[j][k][-1]))

        for s in nfx_ls:
            for f1, f2, f3, f4, f5, f6, f7, f8 in zip(nfx_ls[s], ffd_ls[s], fpd_ls[s], tfd_ls[s],
                                                      mfd_ls[s], fxp_ls[s], nrfx_ls[s], rrdp_ls[s]):
                nfx = np.average(f1)
                ffd = np.average(f2)
                fpd = np.average(f3)
                tfd = np.average(f4)
                mfd = np.average(f5)
                fxp = np.average(f6)
                nrfx = np.average(f7)
                rrdp = np.average(f8)
                self.flat_features.append(np.array([nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]))


class ZuCo2DataNormalizer:
    def __init__(self, dir, task):
        self.dir = dir
        self.task = task

        if self.task == "zuco21":
            self.full_subj = "YAG"  # subject with complete data
        elif self.task.startswith("zuco1"):
            self.full_subj = "ZAB"

        self.files = {}  # dict indexed by subj containing fpaths of corresponding .mat files
        self.subjs = []
        self.flat_words = []
        self.flat_features = []

    def read_files(self):
        LOGGER.info(f"Reading files for task {self.task}")
        for file in sorted(os.listdir(self.dir)):
            if not file.endswith(".mat"):
                continue
            subj = file.split("_")[0][-3:]
            fpath = os.path.join(self.dir, file)
            self.files[subj] = fpath

    def read_subjs(self):
        """
        Reads list of subjs and sorts it such that the full subject is in the first position.
        """
        self.subjs = [subj for subj in self.files]
        for i, subj in enumerate(self.subjs):
            if subj == self.full_subj:
                break
        self.subjs.insert(0, self.subjs.pop(i))

    def read_words(self):
        mat = h5py.File(self.files[self.full_subj])
        sentence_data = mat["sentenceData/word"]
        for row in sentence_data:
            word_content_data = mat[row[0]]["content"]
            for item in word_content_data:
                word = u"".join(chr(c) for c in mat[item[0]].value)
                self.flat_words.append(word)

    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")
        #  dicts indexed by sentence number and word number inside the sentence
        nfx_ls = {}
        ffd_ls = {}
        fpd_ls = {}
        tfd_ls = {}
        mfd_ls = {}
        fxp_ls = {}
        nrfx_ls = {}
        rrdp_ls = {}

        for i, subj in enumerate(self.subjs):
            LOGGER.info(f"Processing subject {i + 1} out of {len(self.subjs)}")
            mat = h5py.File(self.files[subj])
            sentence_data = mat["sentenceData/word"]

            for j, row in enumerate(sentence_data):
                cont = False
                try:
                    cont = np.isnan(mat[row[0]].value[0][0])
                except:
                    pass
                if cont:
                    continue

                word_nfx_data = mat[row[0]]["nFixations"]
                word_ffd_data = mat[row[0]]["FFD"]
                word_gd_data = mat[row[0]]["GD"]
                word_trt_data = mat[row[0]]["TRT"]

                if i == 0:
                    nfx_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    ffd_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    fpd_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    tfd_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    mfd_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    fxp_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    nrfx_ls[j] = [[] for _ in range(len(word_nfx_data))]
                    rrdp_ls[j] = [[] for _ in range(len(word_nfx_data))]

                for k, (i1, i2, i3, i4) in enumerate(zip(word_nfx_data, word_ffd_data, word_gd_data, word_trt_data)):
                    nfx_ls[j][k].append(0 if len(mat[i1[0]].value) == 2 else mat[i1[0]].value[0][0])
                    ffd_ls[j][k].append(0 if len(mat[i2[0]].value) == 2 else mat[i2[0]].value[0][0])
                    fpd_ls[j][k].append(0 if len(mat[i3[0]].value) == 2 else mat[i3[0]].value[0][0])
                    tfd_ls[j][k].append(0 if len(mat[i4[0]].value) == 2 else mat[i4[0]].value[0][0])
                    mfd_ls[j][k].append(get_mean_fix_dur(nfx_ls[j][k][-1], tfd_ls[j][k][-1]))
                    fxp_ls[j][k].append(get_fix_prob(nfx_ls[j][k][-1]))
                    nrfx_ls[j][k].append(get_n_refix(nfx_ls[j][k][-1]))
                    rrdp_ls[j][k].append(get_reread_prob(nrfx_ls[j][k][-1]))

        for s in nfx_ls:
            for f1, f2, f3, f4, f5, f6, f7, f8 in zip(nfx_ls[s], ffd_ls[s], fpd_ls[s], tfd_ls[s],
                                                      mfd_ls[s], fxp_ls[s], nrfx_ls[s], rrdp_ls[s]):
                nfx = np.average(f1)
                ffd = np.average(f2)
                fpd = np.average(f3)
                tfd = np.average(f4)
                mfd = np.average(f5)
                fxp = np.average(f6)
                nrfx = np.average(f7)
                rrdp = np.average(f8)
                self.flat_features.append(np.array([nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]))

class PotsdamDataNormalizer:
    def __init__(self, dir, task):
        self.dir = dir
        self.task = task

        self.files = {}  # dict indexed by frag and subj, each entry is a 2-item list containing the relevant fpaths
        self.frags = []
        self.subjs = []
        self.full_subj = 'reader0'
        self.frags_words = {}  # dict indexed by frag, each entry is the list of words
        self.frags_features = {}  # dict indexed by frag, each entry is the list of numpy arrays containing features
        self.flat_words = []
        self.flat_features = []

    def read_files(self):
        LOGGER.info(f"Reading files for task {self.task}")
        for file in sorted(os.listdir(self.dir)):
            if not file.endswith(".txt"):
                continue
            frag = file.split("_")[2][4:6]
            subj = file.split("_")[1]
            fpath = os.path.join(self.dir, file)
            self.add_file(fpath, frag, subj)

    def add_file(self, fpath, frag, subj):
        if frag not in self.files:
            self.files[frag] = {subj: [fpath]}
        elif subj not in self.files[frag]:
            self.files[frag][subj] = [fpath]
        else:
            self.files[frag][subj].append(fpath)

    def read_frags(self):
        self.frags = [frag for frag in self.files]

    def read_subjs(self):
        self.subjs = [subj for subj in self.files[self.frags[0]]]

    def read_words(self):
        """
        Word list is extracted from the full subject.
        """
        LOGGER.info(f"Reading words for task {self.task}")
        for frag in self.frags:
            self.frags_words[frag] = []
            flt_file = pd.read_csv(self.files[frag][self.full_subj][0], sep=",", header=0)
            for index, row in flt_file.iterrows():
                if index != len(flt_file)-1:
                    if flt_file.at[index+1, 'SentenceBegin'] == 1:
                        self.frags_words[frag].append(str(row['WORD'])+"<eos>")
                    else:
                        self.frags_words[frag].append(str(row['WORD']))
                else:
                    self.frags_words[frag].append(str(row['WORD']) + "<eos>")


    def calc_features(self):
        LOGGER.info(f"Start of features calculation for task {self.task}")

        for i, frag in enumerate(self.frags):
            LOGGER.info(f"Processing fragment {i + 1} out of {len(self.frags)}")
            empty = [[None] * len(self.subjs)] * len(self.frags_words[frag])
            nfx_pd = pd.DataFrame(empty, columns=self.subjs)
            ffd_pd = pd.DataFrame(empty, columns=self.subjs)
            fpd_pd = pd.DataFrame(empty, columns=self.subjs)
            tfd_pd = pd.DataFrame(empty, columns=self.subjs)
            mfd_pd = pd.DataFrame(empty, columns=self.subjs)
            fxp_pd = pd.DataFrame(empty, columns=self.subjs)
            nrfx_pd = pd.DataFrame(empty, columns=self.subjs)
            rrdp_pd = pd.DataFrame(empty, columns=self.subjs)

            for j, subj in enumerate(self.subjs):
                flt_file = pd.read_csv(self.files[frag][self.full_subj][0], sep=",", header=0)
                for k, w in enumerate(self.frags_words[frag]):
                    ffd = flt_file["FFD"].tolist()[k]
                    fpd = flt_file["FPRT"].tolist()[k]
                    tfd = flt_file["TFT"].tolist()[k]
                    nfx = flt_file["nFix"].tolist()[k]
                    mfd = get_mean_fix_dur(nfx, tfd)
                    fxp = get_fix_prob(nfx)
                    nrfx = get_n_refix(nfx)
                    rrdp = get_reread_prob(nrfx)

                    nfx_pd[subj][k] = nfx
                    ffd_pd[subj][k] = ffd
                    fpd_pd[subj][k] = fpd
                    tfd_pd[subj][k] = tfd
                    mfd_pd[subj][k] = mfd
                    fxp_pd[subj][k] = fxp
                    nrfx_pd[subj][k] = nrfx
                    rrdp_pd[subj][k] = rrdp

            nfx = nfx_pd.mean(axis=1).tolist()
            ffd = ffd_pd.mean(axis=1).tolist()
            fpd = fpd_pd.mean(axis=1).tolist()
            tfd = tfd_pd.mean(axis=1).tolist()
            mfd = mfd_pd.mean(axis=1).tolist()
            fxp = fxp_pd.mean(axis=1).tolist()
            nrfx = nrfx_pd.mean(axis=1).tolist()
            rrdp = rrdp_pd.mean(axis=1).tolist()

            features = [nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]
            self.frags_features[frag] = [np.array(i) for i in list(zip(*features))]
            self.flat_words.extend([w for w in self.frags_words[frag]])
            self.flat_features.extend([f for f in self.frags_features[frag]])



class RussSentCorpDataNormalizer:
    def __init__(self, dir, task, print_every=3000):
        self.dir = dir
        self.task = task
        self.print_every = print_every

        self.full_subj = "1.edf"

        self.frags = []
        self.subjs = []
        self.frags_words = {}  # dict indexed by frag, each entry is the list of words
        self.frags_features = {}  # dict indexed by frag, each entry is the list of numpy arrays containing features
        self.flat_words = []
        self.flat_features = []

    def read_file(self):
        LOGGER.info(f"Reading file for task {self.task}")
        self.file = pd.read_csv(self.dir + "/data_103.csv", sep="\t", header=0)


    def read_frags(self):
        self.frags = sorted(self.file["item.id"].unique())
        print(self.frags)
        print(len(self.frags))

    def read_subjs(self):
        """
        Reads list of subjs and sorts it such that the full subject is in the first position.
        """
        LOGGER.info(f"Reading subjects for task {self.task}")
        self.subjs = sorted(self.file["DATA_FILE"].unique())

        print(self.subjs)
        print(len(self.subjs))

    def read_words(self):
        """
        Word list is extracted from the full subject.
        """

        LOGGER.info(f"Reading words for task {self.task}")

        for f in self.frags:
            self.frags_words[f] = []
            flt_file = self.file
            isfrag = flt_file["item.id"] == f
            issubj = flt_file["DATA_FILE"] == self.full_subj
            flt_file = flt_file[isfrag]
            flt_file = flt_file[issubj]
            flt_file["word.serial.no"] = flt_file["word.serial.no"].astype(str).astype(float)
            flt_file = flt_file.sort_values(by=["word.serial.no"])
            sent = [str(w) for w in flt_file["word.id"].tolist()]
            if sent:
                sent[-1] += "<eos>"
            if not self.frags_words[f]:
                self.frags_words[f] = sent
                #print(f, s, [str(w) for w in flt_file["word.id"].tolist()])

        print(len(self.frags_words))

    def calc_features(self):

        LOGGER.info(f"Start of features calculation for task {self.task}")
        for i, frag in enumerate(self.frags):
            LOGGER.info(f"Processing fragment {i + 1} out of {len(self.frags)}")
            empty = [[None] * len(self.subjs)] * len(self.frags_words[frag])
            nfx_pd = pd.DataFrame(empty, columns=self.subjs)
            ffd_pd = pd.DataFrame(empty, columns=self.subjs)
            fpd_pd = pd.DataFrame(empty, columns=self.subjs)
            tfd_pd = pd.DataFrame(empty, columns=self.subjs)
            mfd_pd = pd.DataFrame(empty, columns=self.subjs)
            fxp_pd = pd.DataFrame(empty, columns=self.subjs)
            nrfx_pd = pd.DataFrame(empty, columns=self.subjs)
            rrdp_pd = pd.DataFrame(empty, columns=self.subjs)

            for j, subj in enumerate(self.subjs):
                flt_file = self.file
                isfrag = flt_file["item.id"] == frag
                issubj = flt_file["DATA_FILE"] == subj
                flt_file = flt_file[isfrag]
                flt_file = flt_file[issubj]
                flt_file["word.serial.no"] = flt_file["word.serial.no"].astype(str).astype(float)
                flt_file = flt_file.sort_values(by=["word.serial.no"])

                if len(flt_file["word.id"].to_list()) == len(self.frags_words[frag]):
                    for k, w in enumerate(self.frags_words[frag]):
                        nfx = 0.0 if flt_file["IA_FIXATION_COUNT"].to_list()[k] == "NA" else float(flt_file["IA_FIXATION_COUNT"].to_list()[k])
                        ffd = 0.0 if flt_file["IA_FIRST_FIXATION_DURATION"].to_list()[k] == "NA" else float(flt_file["IA_FIRST_FIXATION_DURATION"].to_list()[k])
                        fpd = 0.0 if flt_file["IA_FIRST_RUN_DWELL_TIME"].to_list()[k] == "NA" else float(flt_file["IA_FIRST_RUN_DWELL_TIME"].to_list()[k])
                        tfd = 0.0 if flt_file["IA_DWELL_TIME"].to_list()[k] == "NA" else flt_file["IA_DWELL_TIME"].to_list()[k]
                        mfd = get_mean_fix_dur(nfx, tfd)
                        fxp = get_fix_prob(nfx)
                        nrfx = get_n_refix(nfx)
                        rrdp = get_reread_prob(nrfx)

                        nfx_pd[subj][k] = nfx
                        ffd_pd[subj][k] = ffd
                        fpd_pd[subj][k] = fpd
                        tfd_pd[subj][k] = tfd
                        mfd_pd[subj][k] = mfd
                        fxp_pd[subj][k] = fxp
                        nrfx_pd[subj][k] = nrfx
                        rrdp_pd[subj][k] = rrdp

            nfx = nfx_pd.mean(axis=1).tolist()
            ffd = ffd_pd.mean(axis=1).tolist()
            fpd = fpd_pd.mean(axis=1).tolist()
            tfd = tfd_pd.mean(axis=1).tolist()
            mfd = mfd_pd.mean(axis=1).tolist()
            fxp = fxp_pd.mean(axis=1).tolist()
            nrfx = nrfx_pd.mean(axis=1).tolist()
            rrdp = rrdp_pd.mean(axis=1).tolist()

            features = [nfx, ffd, fpd, tfd, mfd, fxp, nrfx, rrdp]
            #print(features)
            self.frags_features[frag] = [np.array(i) for i in list(zip(*features))]
            self.flat_words.extend([w for w in self.frags_words[frag]])
            self.flat_features.extend([f for f in self.frags_features[frag]])
            #print(len(self.flat_words))
            #print(len(self.flat_features))
        print(len(self.flat_words))
        print(len(self.flat_features))


class GazeDataNormalizer(DundeeDataNormalizer, GECODataNormalizer, ZuCo1DataNormalizer, ZuCo2DataNormalizer, PotsdamDataNormalizer, RussSentCorpDataNormalizer):
    def __init__(self, dir, task, split_percs):
        if task == "dundee":
            DundeeDataNormalizer.__init__(self, dir, task)
        elif task == "geco" or task == "geco-nl":
            GECODataNormalizer.__init__(self, dir, task)
        elif task == "zuco11" or task == "zuco12":
            # if using the old MATLAB files, change back to ZuCo1DataNormalizer
            ZuCo2DataNormalizer.__init__(self, dir, task)
        elif task == "zuco21":
            ZuCo2DataNormalizer.__init__(self, dir, task)
        elif task == "potsdam":
            PotsdamDataNormalizer.__init__(self, dir, task)
        elif task == "rsc":
            RussSentCorpDataNormalizer.__init__(self, dir, task)

        self.split_percs = split_percs  # percentages of the train, val, and test splits

        self.modes = ["train", "val", "test"]
        self.stopchars = [".", "?", "!", ". ", "? ", "! ", '?"', '."', "!'''", ".'"]  # stopchars that separate sentences
        if task == "potsdam" or task == "rsc":
            self.stopchars = ["<eos>"]

        self.words = {}  # split lists of words
        self.features = {}  # split lists of features

    def read_pipeline(self):
        LOGGER.info(f"Begin loading data for task {self.task}")
        if self.task == "dundee":
            DundeeDataNormalizer.read_files(self)
            DundeeDataNormalizer.read_frags(self)
            DundeeDataNormalizer.read_subjs(self)
            DundeeDataNormalizer.read_words(self)
            DundeeDataNormalizer.calc_features(self)
        elif self.task == "geco" or self.task == "geco-nl":
            GECODataNormalizer.read_file(self)
            GECODataNormalizer.read_frags(self)
            GECODataNormalizer.read_subjs(self)
            GECODataNormalizer.read_words(self)
            GECODataNormalizer.calc_features(self)
        elif self.task == "zuco11" or self.task == "zuco12":
            # if using the old MATLAB file, change back to ZuCo1DataNormalizer
            ZuCo2DataNormalizer.read_files(self)
            ZuCo2DataNormalizer.read_subjs(self)
            ZuCo2DataNormalizer.read_words(self)
            ZuCo2DataNormalizer.calc_features(self)
        elif self.task == "zuco21":
            ZuCo2DataNormalizer.read_files(self)
            ZuCo2DataNormalizer.read_subjs(self)
            ZuCo2DataNormalizer.read_words(self)
            ZuCo2DataNormalizer.calc_features(self)
        elif self.task == "potsdam":
            PotsdamDataNormalizer.read_files(self)
            PotsdamDataNormalizer.read_frags(self)
            PotsdamDataNormalizer.read_subjs(self)
            PotsdamDataNormalizer.read_words(self)
            PotsdamDataNormalizer.calc_features(self)
        elif self.task == "rsc":
            RussSentCorpDataNormalizer.read_file(self)
            RussSentCorpDataNormalizer.read_frags(self)
            RussSentCorpDataNormalizer.read_subjs(self)
            RussSentCorpDataNormalizer.read_words(self)
            RussSentCorpDataNormalizer.calc_features(self)


        self.split()

    def split(self):
        LOGGER.info(f"Splitting the data into sentences for task {self.task}")
        flat_sentences = []
        flat_features = []
        sentence = []
        features = []

        for i in range(len(self.flat_words)):
            sentence.append(self.flat_words[i])
            features.append(self.flat_features[i])

            if sentence[-1].endswith(tuple(self.stopchars)):
                sentence[-1] = sentence[-1].replace("<eos>","")
                flat_sentences.append(sentence)
                flat_features.append(features)
                # sanity check:
                if len(sentence) > 100:
                    print(sentence)
                    print(len(sentence))
                sentence = []
                features = []

        LOGGER.info(f"Splitting data for task {self.task}")

        pairs = list(zip(flat_sentences, flat_features))
        shuffled_pairs = random.sample(pairs, len(pairs))
        len_train = round(self.split_percs["train"] * len(pairs))
        len_val = round(self.split_percs["val"] * len(pairs))

        corpora = {
            "train": shuffled_pairs[:len_train],
            "val": shuffled_pairs[len_train:len_train + len_val],
            "test": shuffled_pairs[len_train + len_val:]
        }
        for mode in self.modes:
            self.words[mode], self.features[mode] = map(list, zip(*corpora[mode]))

    def save_datasets(self):
        LOGGER.info(f"Saving flat dataset for task {self.task}")
        flat_words_pd = pd.DataFrame(self.flat_words, columns=["word"])
        flat_features_pd = pd.DataFrame(self.flat_features, columns=["n_fix", "first_fix_dur", "first_pass_dur",
                                                                     "total_fix_dur", "mean_fix_dur", "fix_prob",
                                                                     "n_refix", "reread_prob"])
        flat_dataset_pd = pd.concat((flat_words_pd, flat_features_pd), axis=1)
        flat_dataset_pd.to_csv(os.path.join(self.dir, "dataset.csv"))

        LOGGER.info(f"Saving split datasets for task {self.task}")
        for mode in self.modes:
            dataset = {
                "sentence_num": [],
                "word": [],
                "n_fix": [],
                "first_fix_dur": [],
                "first_pass_dur": [],
                "total_fix_dur": [],
                "mean_fix_dur": [],
                "fix_prob": [],
                "n_refix": [],
                "reread_prob": []
            }
            for k in range(len(self.words[mode])):
                for j in range(len(self.words[mode][k])):
                    dataset["sentence_num"].append(k)
                    dataset["word"].append(self.words[mode][k][j])
                    dataset["n_fix"].append(self.features[mode][k][j][0])
                    dataset["first_fix_dur"].append(self.features[mode][k][j][1])
                    dataset["first_pass_dur"].append(self.features[mode][k][j][2])
                    dataset["total_fix_dur"].append(self.features[mode][k][j][3])
                    dataset["mean_fix_dur"].append(self.features[mode][k][j][4])
                    dataset["fix_prob"].append(self.features[mode][k][j][5])
                    dataset["n_refix"].append(self.features[mode][k][j][6])
                    dataset["reread_prob"].append(self.features[mode][k][j][7])

            dataset_pd = pd.DataFrame(dataset)
            dataset_pd.to_csv(os.path.join(self.dir, mode + "_dataset.csv"))
