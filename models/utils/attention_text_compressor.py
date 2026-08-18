import torch
import torch.nn as nn

class AttentionCompressor(nn.Module):
    def __init__(self, hidden_dim, num_tokens=10, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ln_output = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        B = hidden_states.size(0)
        hidden_states = self.ln_input(hidden_states)
        queries = self.query_tokens.expand(B, -1, -1)
        encoded, _ = self.attn(query=queries, key=hidden_states, value=hidden_states)
        encoded = self.ln_output(encoded)

        return encoded