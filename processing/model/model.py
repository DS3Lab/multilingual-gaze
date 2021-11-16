import torch
import torch.nn as nn
from transformers import XLMForTokenClassification, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel



class TokenClassificationModel:
    """
    Wrapper for the BERT model for token classification. The init
    method loads the pretrained parameters and stores the init arguments.
    """

    @classmethod
    def init(cls, cf, d_out):
        if "bert" in cf.model_pretrained:
            if cf.random_weights is True:
                # initiate Bert with random weights
                model =  AutoModel.from_config(AutoConfig.from_pretrained(cf.model_pretrained))
            else:
                model = BertForTokenClassification.from_pretrained(cf.model_pretrained, num_labels=d_out,
                                        output_attentions=False, output_hidden_states=False)

        elif "xlm" in cf.model_pretrained:
            model = XLMForTokenClassification.from_pretrained(cf.model_pretrained, num_labels=d_out, output_attentions=False, output_hidden_states=False)



        model.d_out = d_out

        return model


class GazePredictionLoss:
    """
    Loss that deals with a list of variable length sequences. The object call returns global + per-feature MAE loss.
    """

    def __init__(self, d_gaze):
        self.d_gaze = d_gaze
        self.d_report = d_gaze + 1

        self.loss = nn.L1Loss(reduction="sum")

    def __call__(self, b_output, b_target):
        b_length = [len(i) for i in b_output]
        losses = torch.zeros(self.d_report)

        losses[0] = sum([self.loss(i, j) for i, j in zip(b_output, b_target)])
        for output_orig_len, target_orig_len in zip(b_output, b_target):
            for i in range(1, self.d_report):
                losses[i] += self.loss(output_orig_len[:, i - 1], target_orig_len[:, i - 1])

        losses[0] /= sum([i * self.d_gaze for i in b_length])
        losses[1:] /= sum(b_length)
        return losses


def create_finetuning_optimizer(cf, model):
    """
    Creates an Adam optimizer with weight decay. We can choose whether to perform full finetuning on
    all parameters of the model or to just optimize the parameters of the final classification layer.
    """
    if cf.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay_rate": cf.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay_rate": 0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return AdamW(optimizer_grouped_parameters, lr=cf.lr, eps=cf.eps)


def create_scheduler(cf, optim, dl):
    """
    Creates a linear learning rate scheduler.
    """
    n_iters = cf.n_epochs * len(dl)
    return get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=n_iters)
