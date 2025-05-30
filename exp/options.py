import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')


# --------------------
# Experiment Options
# --------------------
parser.add_argument('--mode', type=str, default='train') ## or 'test'
parser.add_argument('--exp_name', type=str, default='DP_CLIP')
parser.add_argument('--model_name', type=str, default='last.ckpt')
parser.add_argument('--FG', type=bool, default=True)
parser.add_argument('--vis', type=bool, default=False)
parser.add_argument('--vis_path', type=str, default='./vis')
parser.add_argument('--vis_rank', type=bool, default=False)
parser.add_argument('--vis_rank_path', type=str, default='./vis_rank')

# ----------------------
# ViT Prompt Parameters
# ----------------------
# parser.add_argument('--total_d_layer', type=int, default=1)
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)
parser.add_argument('--lora', type=bool, default=True)
parser.add_argument('--lora_scale', type=bool, default=True)
parser.add_argument('--rank', type=int, default=16)
parser.add_argument('--prompt', type=str, default='CS_deep')

# ----------------------
# model setting Parameters
# ----------------------
parser.add_argument('--local', type=int, default=4)
parser.add_argument('--local_loss_weight', type=float, default=0.1)
# --------------------
# Loss Options
# --------------------
parser.add_argument('--hard', type=bool, default=True)
parser.add_argument('--margin', type=float, default=0.15)  # margin for triplet loss

# --------------------
# DataLoader Options
# --------------------
# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--dataset', type=str, default='Sketchy') 
parser.add_argument('--data_dir', type=str, default='../datasets/Sketchy/') 
parser.add_argument('--data_split', type=int, default=1)
parser.add_argument('--max_size', type=int, default=224)

parser.add_argument('--train_gray', type=float, default=0.5)
parser.add_argument('--horflip', type=float, default=0.5)
parser.add_argument('--rotation', type=float, default=0.5)
parser.add_argument('--rotation_degree', type=int, default=30)
# ----------------------
# Training Params
# ----------------------
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epoches', type=int, default=60)
parser.add_argument('--freeze_attn', type=bool, default=True) 

# ----------------------
# Val Params
# ----------------------
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)

opts = parser.parse_args()
print(opts)