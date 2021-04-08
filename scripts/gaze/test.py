from settings import DEVICE, reproducibility_setup, RANDOM_STATE

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
    LOGGER.info("Model:", cf.bert_pretrained)

    tokenizer = create_tokenizer(cf.bert_pretrained)

    for task in tasks:

        if args.test_task is None:
            test_task = task
        else:
            test_task = args.test_task

        results_task_dir = os.path.join(results_gaze_dir, task)

        model_init_args = load_json(os.path.join(results_task_dir, "model_init.json"))
        model = TokenClassificationModel.init(cf, **model_init_args)
        if cf.finetune_on_gaze:
            # set finetune_on_gaze to False to test the pretrained models without fine-tuning on eye-tracking data
            LOGGER.info("Fine-tuning on eye-tracking data!")
            model.load_state_dict(torch.load(os.path.join(results_task_dir, "model-"+str(RANDOM_STATE)+".pth")))
        else:
            LOGGER.info("NOT fine-tuning on eye-tracking data!")

        d = GazeDataset(cf, tokenizer, os.path.join(data_gaze_dir, test_task), test_task)
        d.read_pipeline()

        dl = GazeDataLoader(cf, d.numpy["test"], d.target_pad, mode="test")

        tester = GazeTester(model, dl, DEVICE, task)
        tester.evaluate()

        eval_dir = os.path.join(results_gaze_dir, task)
        tester.save_preds(os.path.join(eval_dir, "preds-"+str(RANDOM_STATE)+"-"+cf.bert_pretrained.replace("/","")+".csv"))
        tester.save_logs(os.path.join(eval_dir, "results."+str(RANDOM_STATE)+"-"+cf.bert_pretrained.replace("/","")+".log"))
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
