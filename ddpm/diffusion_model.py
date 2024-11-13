import torch
import torch.nn as nn
import cv2
import numpy as np

class DiffusionModel:
  def __init__(self, model: nn.Module, betas: list[float], num_step: int, device: torch.device):
    if betas is None:
      # torch.linspace(start, end, steps) -> 生成一个从start到end的直线的step次采样
      betas = torch.linspace(1e-4, 0.01, steps=1000)
    self.betas = torch.Tensor(betas).to(device)
    self.alphas = 1 - betas
    # torch.cumprod: 累乘，例如[1,2,3,4,5] -> [1,2=1*2,6=1*2*3,24=1*2*3*4,120=1*2*3*4*5]
    self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
    self.device = device
    self.model = model.to(device)
    self.num_step = num_step
    
  def sample_forward(self, x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor =None) -> torch.Tensor:
    if eps is None:
      eps = torch.randn_like(x_0)
    alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) 
    res = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return res
  
  def sample_backward_and_get_list(self, img_shape, save_step=10):
    imgs = []    
    x = torch.randn(img_shape).to(self.device)
    for t in range(self.num_step - 1, -1, -1):
        x = self.sample_backward_step(x, t)        
        if t % save_step == 0:
          imgs.append(x[0])
    # 最后结果一定要保存      
    imgs.pop()    
    imgs.append(x[0])    
    return imgs
  
  def sample_backward(self, img_shape):
    x = torch.randn(img_shape).to(self.device)
    for t in range(self.num_step - 1, -1, -1):
        x = self.sample_backward_step(x, t)        
    return x
  
  def sample_backward_step(self, x_t: torch.Tensor, t_int: int) -> torch.Tensor:
    N = x_t.shape[0]
    
    t_tensor = torch.full((N,), t_int, device=self.device)
    eps_theta = self.model(x_t, t_tensor)
    
    z = torch.randn_like(x_t) if t_int > 1 else 0
    sigma_t = torch.sqrt(self.betas[t_int])
    
    weighted_eps_theta = (1 - self.alphas[t_int]) / torch.sqrt(1 - self.alpha_bars[t_int]) * eps_theta
    
    mean = 1 / torch.sqrt(self.alphas[t_int]) * (x_t - weighted_eps_theta)
    
    x_t_minus_1 = mean + sigma_t * z
    
    return x_t_minus_1
