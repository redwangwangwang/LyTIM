import torch
import torch.nn as nn
from transformers import XCLIPModel, XCLIPConfig, AutoTokenizer


class XCLIPTextEncoder(nn.Module):
    def __init__(self, pretrained_model='./pretrain_weights/xclip-base-patch32/'):
        super(XCLIPTextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        xclip = XCLIPModel.from_pretrained(pretrained_model)
        self.text_model = xclip.text_model
        self.logit_scale = xclip.logit_scale

    def forward(self, text, device):
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            text_features = self.text_model(**tokens)[1]

        return text_features, self.logit_scale.exp()