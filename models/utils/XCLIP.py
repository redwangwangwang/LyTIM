import torch
import torch.nn as nn
from transformers import XCLIPModel, XCLIPVisionModel, XCLIPConfig

class XCLIPEncoder(nn.Module):
    def __init__(self,
                 num_frames=3,
                 pretrained_model='./pretrain_weights/xclip-base-patch32/'):
        super(XCLIPEncoder, self).__init__()

        config = XCLIPConfig.from_pretrained(pretrained_model)
        config.vision_config.num_frames = num_frames
        config.vision_config.output_attentions = True
        self.model = XCLIPModel.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True).vision_model
        self.hidden_size = self.model.config.hidden_size

    def forward(self, pixel_values, output_attentions=False):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(-1, num_channels, height, width)

        vision_outputs = self.model(pixel_values=pixel_values)
        vision_attentions = vision_outputs.attentions[-1]
        _, num_heads, tgt_len, src_len = vision_attentions.shape
        vision_attentions = vision_attentions.view(batch_size, num_frames, num_heads, tgt_len, src_len)

        output = vision_outputs.last_hidden_state[:, 1:, :]
        output = output.reshape(batch_size, num_frames, -1, self.hidden_size)

        if output_attentions:
            return output, vision_attentions
        else:
            return output
