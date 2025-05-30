import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LORA_Layer(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        # nn.Linear.__init__(self, in_features, out_features, **kwargs)
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        
        
        
        # self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()
        # if fan_in_fan_out:
            # self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def forward(self, x: torch.Tensor, scale=1.0, shift=0.0):

        mid_result = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
        mid_result = mid_result + mid_result * scale + shift * 0.1
        
        result = (mid_result @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    