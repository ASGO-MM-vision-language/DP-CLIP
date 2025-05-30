import torch
import torch.nn as nn
from torch.nn import Dropout

import math 
from operator import mul
from functools import reduce

from .functions import init_weights
from .clip.model import LayerNorm,ResidualAttentionBlock

import pdb

class Prompt(nn.Module):
    def __init__(self, opts, patch_size, width):
        super().__init__()

        self.total_d_layer = 12
        
        self.n_prompts = opts.n_prompts
        print('prompt:', opts.prompt, 'layer:', self.total_d_layer)

        prompt_dim = opts.prompt_dim
        # if opts.prompt != 'text_vist_shallow' and opts.prompt != 'text_vist_deep':
        self.prompt_embeddings = self.init_prompts(patch_size, self.n_prompts, prompt_dim, self.total_d_layer)
        print('self.prompt_embeddings: ', self.prompt_embeddings.shape)

        ## prompt related functions
        self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
        self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
        self.prompt_dropout = Dropout(0.1)

        
        heads = width // 64
        self.prompt_predictor = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask=None) for _ in range(1)])
        # ResidualAttentionBlock(width, heads, attn_mask=None)
        self.prompt_predictor.apply(init_weights)


    def init_prompts(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        
        prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(prompt_embeddings.data, -val, val)
        return prompt_embeddings


    ## CS visual ptompt
    def generate_cate_adapt_prompt(self, trips):  
        inputs = torch.unsqueeze(trips, 0) 
        
        if self.total_d_layer == 1:
            x = torch.cat([inputs, self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))], dim=1)
            x = x.permute(1, 0, 2)  ## [1, 50*3+n_prompts, 768] --> [153, 1, 768]
            x = self.prompt_predictor(x)
            x = x.permute(1, 0, 2) 
            hidden_features = x[:, inputs.shape[1]:(inputs.shape[1] + self.n_prompts)] #[1,nn_prompts, 768]
            return hidden_features

        hidden_features = []

        for i in range(self.total_d_layer):
            prompt_embeds = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[i]))
            x = torch.cat([inputs, torch.unsqueeze(prompt_embeds, 0)], dim=1)
            x = x.permute(1, 0, 2)  ## [1, 153, 768] --> [153, 1, 768]
            x = self.prompt_predictor(x)
            x = x.permute(1, 0, 2)
            # hidden_features.append(x[:, :self.n_prompts]) 
            hidden_features.append(x[:, inputs.shape[1]:(inputs.shape[1] + self.n_prompts)]) 

        hidden_features = torch.stack(hidden_features)  ### torch.Size([12, 64, 3, 768])

        return hidden_features


    ## common ptompt
    def generate_common_prompt(self):  
        
        if self.total_d_layer == 1:
            x = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))
            # x = x.permute(1, 0, 2)  ## [1, n_prompts, 768] --> [3, 1, 768]
            # x = self.prompt_predictor(x)
            # x = x.permute(1, 0, 2) 
            hidden_features = x  #[1,nn_prompts, 768]
            return hidden_features

        hidden_features = []

        for i in range(self.total_d_layer):
            prompt_embeds = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[i]))
            x = torch.unsqueeze(prompt_embeds, 0)
            # x = x.permute(1, 0, 2)  ## [1, 153, 768] --> [153, 1, 768]
            # x = self.prompt_predictor(x)
            # x = x.permute(1, 0, 2)
            # hidden_features.append(x[:, :self.n_prompts]) 
            hidden_features.append(x) 

        hidden_features = torch.stack(hidden_features)  ### torch.Size([12, 64, 3, 768])

        return hidden_features

