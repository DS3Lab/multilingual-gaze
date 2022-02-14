from settings import DEVICE, reproducibility_setup, RANDOM_STATE
from transformers import AutoConfig

reproducibility_setup()

import argparse
import os
import torch

from processing import TokenClassificationModel, GazeTester, GazeDataLoader, Config, GazeDataset, \
    create_tokenizer, load_json, LOGGER


def main(args):
    data_gaze_dir = args.data_gaze_dir
    results_gaze_dir = args.results_gaze_dir
    tasks = args.tasks

    cf = Config.load_json(os.path.join(results_gaze_dir, "config.json"))

    tokenizer = create_tokenizer(cf.model_pretrained)

    for task in tasks:

        if args.test_task is None:
            test_task = task
        else:
            test_task = args.test_task

        results_task_dir = os.path.join(results_gaze_dir, task)

        model_init_args = load_json(os.path.join(results_task_dir, "model_init.json"))

        LOGGER.info("initiating random Bert model: ")
        LOGGER.info(cf.random_weights)
        model = TokenClassificationModel.init(cf, **model_init_args)

        if not cf.random_baseline:
            # set random_baseline to True in the cf file loaded above to test on a randomly initialized regression
            LOGGER.info("Fine-tuned on eye-tracking data!")
            LOGGER.info("model-"+cf.model_pretrained+"-"+str(cf.full_finetuning)+"-"+str(RANDOM_STATE)+".pth")
            model.load_state_dict(torch.load(os.path.join(results_task_dir, "model-"+cf.model_pretrained+"-"+str(cf.full_finetuning)+"-"+str(RANDOM_STATE)+".pth")))
            print(model.classifier.weight.data)
        else:
            LOGGER.info("Random regression layer, NO trained weights loaded!")
            print(model.classifier.weight.data)

        d = GazeDataset(cf, tokenizer, os.path.join(data_gaze_dir, test_task), test_task)
        d.read_pipeline()

        dl = GazeDataLoader(cf, d.numpy["test"], d.target_pad, mode="test")

        #LOGGER.info(model)

        tester = GazeTester(model, dl, DEVICE, task)
        tester.evaluate()

        eval_dir = os.path.join(results_gaze_dir, task)
        tester.save_preds(os.path.join(eval_dir, "preds-"+str(RANDOM_STATE)+"-"+cf.model_pretrained.replace("/","")+"-"+str(cf.full_finetuning)+"-"+str(cf.random_weights)+"-"+str(cf.random_baseline)+".csv"))
        tester.save_logs(os.path.join(eval_dir, "results."+str(RANDOM_STATE)+"-"+cf.model_pretrained.replace("/","")+"-"+str(cf.full_finetuning)+"-"+str(cf.random_weights)+"-"+str(cf.random_baseline)+".log"))
        tester.save_logs_all(os.path.join(results_gaze_dir, "result_log.csv"), RANDOM_STATE, cf)
        LOGGER.info(f"Testing completed, training on task {task}, testing on {test_task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dg", "--data_gaze_dir", type=str,
                        default="../../data/gaze")
    parser.add_argument("-rg", "--results_gaze_dir", type=str,
                        default="../../results/gaze")
    parser.add_argument("-ts", "--tasks", type=str, nargs="+",
                        default=["zuco"])
    parser.add_argument("-tt", "--test_task", type=str, default=None)
    args = parser.parse_args()
    main(args)
