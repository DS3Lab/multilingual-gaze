import json

import mlflow

from processing.utils.utils import save_json


class Config:
    def save_json(self, fpath):
        save_json(self.__dict__, fpath)

    @classmethod
    def load_json(cls, fpath):
        cf = cls()
        with open(fpath) as f:
            cf.__dict__ = json.load(f)

        return cf

    def log_mlflow_params(self):
        mlflow.log_params(self.__dict__)
