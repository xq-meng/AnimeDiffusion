import json
import torch
from torchvision import datasets, transforms
from models.diffusion import GaussianDiffusion

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    # dataset MNIST
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussian_diffusion = GaussianDiffusion(config['model'])
    gaussian_diffusion.to(device)
    optimizer = torch.optim.Adam(gaussian_diffusion.denoise_fn.parameters(), lr=5e-4)

    epochs = 10

    for epoch in range(epochs):
        print("Epoch: {0}".format(epoch))
    
        for step, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            batch_size = images.shape[0]
            images = images.to(device)
            t = torch.randint(0, config['model']['time_step'], (batch_size, ), device=device).long()
            loss = gaussian_diffusion.train(images, t)
            if step % 200 == 0:
                print("Loss: {0}".format(loss.item()))
        
            loss.backward()
            optimizer.step()