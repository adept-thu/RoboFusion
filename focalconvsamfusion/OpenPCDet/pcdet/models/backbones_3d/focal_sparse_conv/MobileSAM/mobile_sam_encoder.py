import torch
import torch.nn as nn
from mobile_sam import image_encoder_registry
from torchvision.transforms.functional import resize
from torch.nn import functional as F



class MobileSamImageEncoder(nn.Module):
    def __init__(self, encoder_checkpoint, model_type='vit_t', im_size=1024, is_optimize=False):
        super().__init__()
        self.im_size = im_size
        self.is_optimize = is_optimize
        self.image_encoder = image_encoder_registry[model_type](checkpoint=encoder_checkpoint)

    def preprocess(self, x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size = 1024) -> torch.Tensor:
        """
        For sam_encoder
        Normalize pixel values and resize to a square input.
        """
        
        assert x.shape[1] == 3, f"The batch images should be 3 for RGB, but get {x.shape[1]}."
        x = x * 255
        x = resize(x, (img_size, img_size))
        
        
        # Normalize colors
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        pixel_mean = torch.as_tensor(pixel_mean, device="cuda")
        pixel_std = torch.as_tensor(pixel_std, device="cuda")
        x = (x - pixel_mean) / pixel_std
        
        # # Pad
        # h, w = x.shape[-2:]
        # padh = img_size - h
        # padw = img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def forward(self, x_rgb):
        x_rgb = self.preprocess(x_rgb)
        x_rgb = self.image_encoder(x_rgb)
        if not self.is_optimize:
            x_rgb.detach_()
        return x_rgb