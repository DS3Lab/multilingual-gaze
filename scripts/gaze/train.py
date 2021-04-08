from settings import DEVICE, mlflow_setup, reproducibility_setup

reproducibility_setup()

import argparse
import os

import mlflow

from processing import TokenClassificationModel, GazeTrainer, GazeDataLoader, Config, GazeDataset, \
    create_finetuning_optimizer, create_scheduler, create_tokenizer, save_json, LOGGER


def main(args):
    data_gaze_dir = args.data_gaze_dir
    results_gaze_dir = args.results_gaze_dir
    params_gaze_dir = args.params_gaze_dir
    mlflow_dir = args.mlflow_dir
    tasks = args.tasks
    config = args.config

    cf = Config.load_json(os.path.join(params_gaze_dir, config))
    cf.save_json(os.path.join(results_gaze_dir, "config.json"))

    tokenizer = create_tokenizer(cf.model_pretrained)

    for task in tasks:
        d = GazeDataset(cf, tokenizer, os.path.join(data_gaze_dir, task), task)
        d.read_pipeline()

        train_dl = GazeDataLoader(cf, d.numpy["train"], d.target_pad, mode="train")
        val_dl = GazeDataLoader(cf, d.numpy["val"], d.target_pad, mode="val")

        model_init_args = {
            "d_out": d.d_out
        }
        save_json(model_init_args, os.path.join(results_gaze_dir, task, "model_init.json"))

        model = TokenClassificationModel.init(cf, **model_init_args)
        optim = create_finetuning_optimizer(cf, model)
        scheduler = create_scheduler(cf, optim, train_dl)

        mlflow_setup(mlflow_dir)
        mlflow.set_experiment(f"gaze_{task}")
        with mlflow.start_run():
            cf.log_mlflow_params()

            eval_dir = os.path.join(results_gaze_dir, task)
            trainer = GazeTrainer(cf, model, train_dl, val_dl, optim, scheduler, eval_dir, task,
                                  DEVICE, monitor="loss_all", monitor_mode="min")
            trainer.train()

        LOGGER.info(f"Training completed task {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dg", "--data_gaze_dir", type=str,
                        default="../../data/gaze")
    parser.add_argument("-rg", "--results_gaze_dir", type=str,
                        default="../../results/gaze")
    parser.add_argument("-pg", "--params_gaze_dir", type=str,
                        default="../..params/gaze")
    parser.add_argument("-md", "--mlflow_dir", type=str,
                        default="../../mlruns")
    parser.add_argument("-ts", "--tasks", type=str, nargs="+",
                        default=["zuco"])
    parser.add_argument("-cf", "--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args)
