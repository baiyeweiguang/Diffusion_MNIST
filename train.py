import torch
import torch.nn as nn
from ddpm import DiffusionModel
from models import UNet
from minist_dataset import get_dataloader
import cv2
import os
import einops
import numpy as np
from tqdm import tqdm

def train(ddpm: DiffusionModel, device, ckpt_path):
  batch_size = 512
  n_epochs = 100
  
  n_steps = ddpm.num_step
  train_set, val_set = get_dataloader(batch_size)
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(ddpm.model.parameters(), 1e-3)

  for e in range(n_epochs):
    
    ddpm.model.train()
    loop = tqdm(train_set, total=len(train_set), leave=False)
    
    for i, (x, _) in enumerate(loop):
      batch_size = x.shape[0]
      x = x.to(device)
      t = torch.randint(0, n_steps, (batch_size, )).to(device)
      eps = torch.randn_like(x).to(device)
      x_t = ddpm.sample_forward(x, t, eps)
      eps_theta = ddpm.model(x_t, t)
      loss = loss_fn(eps_theta, eps)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loop.set_description(f'Epoch {e}, Loss: {loss.item()}')

    print(f'Epoch {e}, Loss: {loss.item()}')
    sample_imgs(ddpm, f'runs/epoch_{e}.jpg', n_sample=25, device=device)
  torch.save(ddpm.model.state_dict(), ckpt_path)

    
def sample_imgs(ddpm,
                output_path,
                n_sample=81,
                device='cuda'):
  ddpm.model.eval()
  with torch.no_grad():
    shape = (n_sample, 1, 28, 28)  # 1, 3, 28, 28
    imgs = ddpm.sample_backward(shape).detach().cpu()
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs,
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',
                            b1=int(n_sample**0.5))

    imgs = imgs.numpy().astype(np.uint8)
    cv2.imwrite(output_path, imgs)
    
if __name__ == '__main__':
    os.makedirs('runs', exist_ok=True)
    ckpt_path = 'runs/model.ckpt'

    device = torch.device('cuda')

    net = UNet()
    ddpm = DiffusionModel(net, None, num_step=1000, device=device)

    train(ddpm, device=device, ckpt_path=ckpt_path)

    net.load_state_dict(torch.load(ckpt_path))
    sample_imgs(ddpm, 'runs/diffusion.jpg', device=device)    
    
    # imgs = ddpm.sample_backward_and_get_list((1, 1, 28, 28))
    # len_ = len(imgs)
    # # list to stacked nparray
    # print(imgs[0].shape)
    # imgs = torch.stack(imgs, axis=0)
    # imgs = (imgs + 1) / 2 * 255
    # imgs = imgs.clamp(0, 255)
    # imgs = einops.rearrange(imgs,
    #                         '(b1 b2) c h w -> (b1 h) (b2 w) c',
    #                         b1=int(len_**0.5))    
    # imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    # cv2.imwrite('steps.jpg', imgs)