""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import *


class UNet(nn.Module):
    def __init__(self, emb_dim:int =128, bilinear=False):
        super(UNet, self).__init__()
        self.channels = [1, 16, 32, 64, 128] #, 256]
        self.n_channels = 1
        self.n_classes = 1
        self.bilinear = bilinear
        self.emb_dim = emb_dim

        self.embed1 = TemporalEmbedding(emb_dim, self.channels[0])
        self.inc = (DoubleConv(self.channels[0], self.channels[1]))
        
        self.embed2 = TemporalEmbedding(emb_dim, self.channels[1])
        self.down1 = (Down(self.channels[1], self.channels[2]))
        
        self.embed3 = TemporalEmbedding(emb_dim, self.channels[2])
        self.down2 = (Down(self.channels[2], self.channels[3]))
        
        self.embed4 = TemporalEmbedding(emb_dim, self.channels[3])
        self.down3 = (Down(self.channels[3], self.channels[4]))
        
        factor = 2 if bilinear else 1
        # self.down4 = (Down(self.channels[4], self.channels[5] // factor))
        
        self.up1 = (Up(self.channels[4], self.channels[3] // factor, bilinear))
        self.embed5 = TemporalEmbedding(emb_dim, self.channels[3])
        
        self.up2 = (Up(self.channels[3], self.channels[2] // factor, bilinear))
        self.embed6 = TemporalEmbedding(emb_dim, self.channels[2])
        
        self.up3 = (Up(self.channels[2], self.channels[1] // factor, bilinear))
        self.embed7 = TemporalEmbedding(emb_dim, self.channels[1])
        # self.up4 = (Up(self.channels[2], self.channels[1], bilinear))
        
        self.outc = (OutConv(self.channels[1], self.channels[0]))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.embed1(x, t)
        x1 = self.inc(x1)
        
        x2 = self.embed2(x1, t)
        x2 = self.down1(x2)
        
        x3 = self.embed3(x2, t)
        x3 = self.down2(x3)
        
        x4 = self.embed4(x3, t)
        x4 = self.down3(x4)
        
        # x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.embed5(x, t)
        
        x = self.up2(x, x2)
        x = self.embed6(x, t)
        
        x = self.up3(x, x1)
        x = self.embed7(x, t)
        
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits      