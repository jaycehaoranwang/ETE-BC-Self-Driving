
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def estimate_depth(img, encoder='vits', weight_path=None):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
        # Determine weights path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script directory
    model_weight_path = os.path.join(script_dir, f'depth_anything_v2_{encoder}.pth')
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model.infer_image(img)
