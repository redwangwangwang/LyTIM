import torch
import torch.nn as nn

class TripletEncoder(nn.Module):
    def __init__(self,
                 llama_embed_dim=4096,
                 chexbert_dim=14,
                 num_prompt_tokens=8):
        super(TripletEncoder, self).__init__()
        self.chexbert_dim = chexbert_dim
        self.llama_embed_dim = llama_embed_dim

        self.init_proj = nn.Linear(chexbert_dim, 128)
        self.init_layernorm = nn.LayerNorm(128)
        self.linear_1 = nn.Linear(256, 512)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(512)
        self.linear_2 = nn.Linear(512, llama_embed_dim * num_prompt_tokens)
        self.final_layer_norm = nn.LayerNorm(llama_embed_dim)

        self.base_prompt_embedding = nn.Parameter(torch.randn(num_prompt_tokens, llama_embed_dim))

    def forward(self, gt_label, pred_label):
        gt_proj = self.init_layernorm(self.init_proj(gt_label.to(dtype=torch.bfloat16)))
        pred_proj = self.init_layernorm(self.init_proj(pred_label.to(dtype=torch.bfloat16)))

        x = torch.cat((gt_proj, pred_proj), dim=1)

        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.layernorm(x)

        x = self.linear_2(x)
        x = x.view(x.shape[0], -1, self.llama_embed_dim)
        x = x + self.base_prompt_embedding.unsqueeze(0)

        x = self.final_layer_norm(x)

        return x