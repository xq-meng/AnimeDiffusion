import json
import torch
from models.diffusion import GaussianDiffusion

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    net = GaussianDiffusion(config['model'])
    x = torch.randint(0, 255, (64, 64))
    net(x)