import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pcdet.models.backbones_3d.focal_sparse_conv.torch_wavelet import DWT_2D, IDWT_2D
from pcdet.models.backbones_3d.focal_sparse_conv.wavevit import Fastwaveblock,WaveAttention1

FusionModel = Fastwaveblock(dim=16)
demo_in = torch.rand(1, 64*64, 16)
print(FusionModel(demo_in,H=64,W=64).shape)