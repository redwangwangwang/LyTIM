from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
import os
import torch
import torch.nn as nn

class CheXbert(nn.Module):
    def __init__(self, checkpoint_path, p=0.1):
        super(CheXbert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('./pretrain_weights/bert-base-uncased/')
        config = BertConfig().from_pretrained('./pretrain_weights/bert-base-uncased/')

        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint
            # state_dict = torch.load(checkpoint_path, map_location=self.device)['model_state_dict']
            state_dict = torch.load(checkpoint_path)['model_state_dict']

            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict, strict=False)

        self.eval()

    def forward(self, reports, return_logits=False):
        device = self.bert.embeddings.position_ids.device

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\s+", " ")
            reports[i] = reports[i].replace("\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():
            tokenized =self.tokenizer(reports,
                                      padding=True,
                                      truncation=True,
                                      max_length=512,
                                      return_tensors='pt')
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            # predictions = []
            # for i in range(14):
            #     predictions.append(self.linear_heads[i](cls).argmax(dim=1))

            predictions = []
            logits = []
            for i in range(14):
                disease_logit = self.linear_heads[i](cls)
                if return_logits:
                    disease_logit = torch.softmax(disease_logit, dim=-1)
                    if i == 13:
                        disease_logit = torch.cat(
                            [
                                disease_logit,
                                torch.zeros_like(disease_logit, dtype=disease_logit.dtype).to(disease_logit.device)
                            ],
                            dim=-1
                        )
                    logits.append(disease_logit)
                else:
                    predictions.append(disease_logit.argmax(dim=1))

        if return_logits:
            return torch.stack(logits, dim=1)
        else:
            # Nx14
            return torch.stack(predictions, dim=1)