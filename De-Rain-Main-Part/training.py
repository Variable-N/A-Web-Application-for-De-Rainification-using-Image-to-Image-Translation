import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import RainDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.cuda.empty_cache()

def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = RainDataset("dataset\_train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):

        loop = tqdm(train_loader, leave=True)

        for _, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())
                D_real_loss = BCE(D_real, torch.ones_like(D_real)) 
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake)) 
                D_loss = (D_real_loss + D_fake_loss) / 2 
        
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
        
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                L1 = L1_LOSS(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1 
        
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
           

        if epoch % 5 == 0:
            torch.save(gen, f'model_weights/gen_{_}.pth')
            torch.save(disc, f'model_weights/disc_{_}.pth')


if __name__ == '__main__':
    main()
