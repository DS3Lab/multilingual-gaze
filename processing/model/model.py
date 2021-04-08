import torch
import torch.nn as nn
from transformers import BertConfig, BertForTokenClassification, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import XLMConfig, XLMModel, XLMForTokenClassification
from processing.utils.utils import n_modules


class LSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


class EnsembleTokenClassificationModel(nn.Module):
    """
    Model for token classification using gaze features and word tokens combined. It's an ensemble of two
    BERT models, where the one for text is pretrained and the one for gaze is small and trained from scratch.
    The gaze BERT features a special embedding layer, to accomodate for a multi-dimensional input.
    """

    def __init__(self, cf, baseline, d_out, d_gaze, gaze_vocab_size, gaze_pad_idx):
        super().__init__()

        self.gaze_mixing_perc = cf.gaze_mixing_perc
        self.baseline = baseline
        self.d_out = d_out

        self.text_bert = BertForTokenClassification.from_pretrained(cf.bert_pretrained,
                                                                    num_labels=d_out,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False)
        if not baseline:
            self.embeddings = nn.ModuleList(n_modules(d_gaze, nn.Embedding, gaze_vocab_size,
                                                      cf.hidden_size // d_gaze, padding_idx=gaze_pad_idx))
            self.gaze_bert = BertForTokenClassification(BertConfig(num_labels=d_out,
                                                                   hidden_size=cf.hidden_size,
                                                                   num_hidden_layers=cf.n_hidden_layers,
                                                                   num_attention_heads=cf.n_attention_heads,
                                                                   intermediate_size=cf.intermediate_size))

    def forward(self, text_input_ids, attention_mask, gaze_input_ids=None):
        logits = self.text_bert(input_ids=text_input_ids, attention_mask=attention_mask)[0]

        if not self.baseline and gaze_input_ids is not None:
            gaze_inputs_embeds = []
            for k, embedding in enumerate(self.embeddings):
                ids = torch.cat([i[:, k].unsqueeze(0) for i in gaze_input_ids], dim=0)
                gaze_inputs_embeds.append(embedding(ids))

            gaze_inputs_embeds = torch.cat(gaze_inputs_embeds, dim=2)

            gaze_logits = self.gaze_bert(inputs_embeds=gaze_inputs_embeds, attention_mask=attention_mask)[0]
            logits = self.gaze_mixing_perc * gaze_logits + (1 - self.gaze_mixing_perc) * logits

        return logits


class EnsembleSequenceClassificationModel(nn.Module):
    """
    Model for sequence classification using gaze features and word tokens combined. It's an ensemble of two
    BERT models, where the one for text is pretrained and the one for gaze is small and trained from scratch.
    The gaze BERT features a special embedding layer, to accomodate for a multi-dimensional input.
    """

    def __init__(self, cf, baseline, d_out, d_gaze, gaze_vocab_size, gaze_pad_idx):
        super().__init__()

        self.gaze_mixing_perc = cf.gaze_mixing_perc
        self.baseline = baseline
        self.d_out = d_out
        self.gaze_pad_idx = gaze_pad_idx

        self.text_bert = BertForSequenceClassification.from_pretrained(cf.bert_pretrained,
                                                                       num_labels=d_out,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)
        if not baseline:
            self.embeddings = nn.ModuleList(n_modules(d_gaze, nn.Embedding, gaze_vocab_size,
                                                      cf.hidden_size // d_gaze, padding_idx=gaze_pad_idx))
            self.gaze_bert = BertForSequenceClassification(BertConfig(num_labels=d_out,
                                                                      hidden_size=cf.hidden_size,
                                                                      num_hidden_layers=cf.n_hidden_layers,
                                                                      num_attention_heads=cf.n_attention_heads,
                                                                      intermediate_size=cf.intermediate_size))

    def forward(self, text_input_ids, attention_mask, gaze_input_ids=None):
        logits = self.text_bert(input_ids=text_input_ids, attention_mask=attention_mask)[0]

        if not self.baseline and gaze_input_ids is not None:
            gaze_inputs_embeds = []
            for k, embedding in enumerate(self.embeddings):
                ids = torch.cat([i[:, k].unsqueeze(0) for i in gaze_input_ids], dim=0)
                gaze_inputs_embeds.append(embedding(ids))

            gaze_inputs_embeds = torch.cat(gaze_inputs_embeds, dim=2)

            gaze_logits = self.gaze_bert(inputs_embeds=gaze_inputs_embeds, attention_mask=attention_mask)[0]
            logits = self.gaze_mixing_perc * gaze_logits + (1 - self.gaze_mixing_perc) * logits

        return logits


class TokenClassificationModel(BertForTokenClassification):
    """
    Wrapper for the BERT model for token classification. The init
    method loads the pretrained parameters and stores the init arguments.
    """

    @classmethod
    def init(cls, cf, d_out):
        if "bert" in cf.bert_pretrained:
            model = super().from_pretrained(cf.bert_pretrained, num_labels=d_out,
                                        output_attentions=False, output_hidden_states=False)
        else:
            model = XLMForTokenClassification.from_pretrained(cf.bert_pretrained, num_labels=d_out, output_attentions=False, output_hidden_states=False)
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
