# Eye-Tracking Feature Prediction
This repo contains the code for the experiments presented in the following paper:  
Nora Hollenstein, Federico Pirovano, Lena Jäger, Ce Zhang & Lisa Beinborn. "Multilingual language models predict human reading behavior". _NAACL 2021_.


## Folders
The repository is divided into three folders:  
* `processing` contains the python package of the project;
* `params` contains example parameter configurations for training and testing;
* `scripts` contains the python runnable scripts of the project.

## Requirements

The Python version used is 3.7.7.

## Running the code

A maximum of three folders have to be specified:
* `--data_gaze_dir` will contain the folders for each dataset. The script expects this folder to contain one dataset folder for each task specified in the arguments. The name of the folder and the name of the task need to match;
* `--results_gaze_dir` will be created to contain the results of the gaze prediction task of the project;
* `--params_gaze_dir` will contain the json configuration files for the gaze prediction task.

The runnable scripts are: [combine_datasets.py](https://github.com/DS3Lab/multilingual-gaze/blob/main/scripts/gaze/combine_datasets.py), [normalize_data.py](https://github.com/DS3Lab/multilingual-gaze/blob/main/scripts/gaze/normalize_data.py), [train.py](https://github.com/DS3Lab/multilingual-gaze/blob/main/scripts/gaze/train.py), and [test.py](https://github.com/DS3Lab/multilingual-gaze/blob/main/scripts/gaze/test.py).  
`combine_datasets.py` combines the datasets passed as option into one dataset under the name `all`.  
The other scripts are meant to be run in order: first `normalize_data.py`, then `train.py`, and finally `test.py`.  
The `normalize_data.py` script will convert and save the data from dataset-specific format into a shared format that can be read by the `GazeDataset` class.

Example:  
`python scripts/gaze/normalize_data.py --data_gaze_dir data/gaze/ --tasks geco-nl`

`python scripts/gaze/combine_datasets.py --data_gaze_dir data/gaze/ --tasks dundee geco zuco-all --percentage 0.8`

`python scripts/gaze/train.py --data_gaze_dir data/gaze/ --results_gaze_dir results/gaze/ --tasks geco-nl --mlflow_dir mlruns/ --params_gaze_dir params/gaze/dutch/`

`python scripts/gaze/test.py --data_gaze_dir data/gaze/ --results_gaze_dir results/gaze/ --tasks geco-nl`


The eye-tracking features are predicted in the following order: n_fix, first_fix_dur, first_pass_dur, total_fix_dur, mean_fix_dur, fix_prob, n_refix, reread_prob.

## Data
The `GazeDataNormalizer` class supports the following datasets:

### English
* Dundee corpus: place all the files in a dataset-specific folder;
* [GECO](http://expsy.ugent.be/downloads/geco/) corpus: download only the `MonolingualReadingData.xlsx` file and place it in a dataset-specific folder;
* [ZuCo 1.0](https://osf.io/q3zws/) corpus: download only the MATLAB files for tasks 1 and 2 and place them in two separated dataset-specific folders;
* [ZuCo 2.0](https://osf.io/2urht/) corpus: download only the MATLAB files for task 1 and place it in a dataset-specific folder. After normalizing the ZuCo data seprately per task, you can use the script `combine_datasets.py` to merge them into a single dataset, if needed.

### Dutch
* [GECO](http://expsy.ugent.be/downloads/geco/) corpus: download only the `L1ReadingData.xlsx` file and place it in a dataset-specific folder;

### German
* [Potsdam Textbook Corpus](https://osf.io/dn5hp/): These files nees to be preprocessed so that the words and corresponding eye-tracking features are merged into one file per text and reader.

### Russian
* [Russian Sentence Corpus](https://osf.io/x5q2r/): Download the file `data_103.csv` and place it in a dataset folder.

The random state can be set in `settings.py`.
The model parameters are set in `params/gaze/config.json`.

