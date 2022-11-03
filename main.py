import json
import torch
from models.diffusion import GaussianDiffusion

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    net = GaussianDiffusion(config['model'])