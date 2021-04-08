import os
import mlflow
import torch

RANDOM_STATE = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reproducibility_setup():
    import random
    random.seed(RANDOM_STATE)

    import numpy as np
    np.random.seed(RANDOM_STATE)

    import torch
    torch.manual_seed(RANDOM_STATE)


def mlflow_setup(mlflow_dir):
    if not os.path.exists(mlflow_dir):
        os.makedirs(mlflow_dir)

    mlflow.set_tracking_uri("file:" + mlflow_dir)
