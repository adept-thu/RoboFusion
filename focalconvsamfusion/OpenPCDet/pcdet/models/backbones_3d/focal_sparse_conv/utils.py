import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from timm.models.layers import DropPath


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).to(index.device)

        index = index.view(*view)
        ones = 1.

        if isinstance(index, Variable):
            ones = Variable(torch.Tensor(index.size()).fill_(1).to(index.device))
            mask = Variable(mask, volatile=index.volatile)

        return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

def sort_by_indices(features, indices, features_add=None):
    """
        To sort the sparse features with its indices in a convenient manner.
        Args:
            features: [N, C], sparse features
            indices: [N, 4], indices of sparse features
            features_add: [N, C], additional features to sort
    """
    idx = indices[:, 1:]
    idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() + idx.select(1, 1) * idx[:, 2].max() + idx.select(1, 2)
    _, ind = idx_sum.sort()
    features = features[ind]
    indices = indices[ind]
    if not features_add is None:
        features_add = features_add[ind]
    return features, indices, features_add

def sort_by_indices2(features, indices, features_add=None):
    """
        To sort the sparse features with its indices in a convenient manner.
        Args:
            features: [N, C], sparse features
            indices: [N, 4], indices of sparse features
            features_add: [N, C], additional features to sort
    """
    idx = indices
    idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() * idx[:, 3].max() + idx.select(1, 1) * idx[:, 2].max() * idx[:, 3].max() + idx.select(1, 2) * idx[:, 3].max() + idx.select(1, 3)
    _, ind = idx_sum.sort()
    features = features[ind]
    indices = indices[ind]
    if not features_add is None:
        features_add = features_add[ind]
    return features, indices, features_add

def check_repeat(features, indices, features_add=None, sort_first=True, flip_first=True):
    """
        Check that whether there are replicate indices in the sparse features, 
        remove the replicate features if any.
    """
    if sort_first:
        features, indices, features_add = sort_by_indices(features, indices, features_add)

    if flip_first:
        features, indices = features.flip([0]), indices.flip([0])

    if not features_add is None:
        features_add=features_add.flip([0])

    idx = indices[:, 1:].int()
    idx_sum = torch.add(torch.add(idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max(), idx.select(1, 1) * idx[:, 2].max()), idx.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)
    
    if _unique.shape[0] < indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
        features_new.index_add_(0, inverse.long(), features)
        features = features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        indices = indices[perm_].int()

        if not features_add is None:
            features_add_new = torch.zeros((_unique.shape[0],), device=features_add.device)
            features_add_new.index_add_(0, inverse.long(), features_add)
            features_add = features_add_new / counts
    return features, indices, features_add

def check_repeat2(features, indices, features_add=None, sort_first=True, flip_first=True):
    """
        Check that whether there are replicate indices in the sparse features, 
        remove the replicate features if any.
    """
    if sort_first:
        features, indices, features_add = sort_by_indices2(features, indices, features_add)

    if flip_first:
        features, indices = features.flip([0]), indices.flip([0])

    if not features_add is None:
        features_add=features_add.flip([0])

    idx = indices #[:, 1:].int()
    #idx_sum = torch.add(torch.add(idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max(), idx.select(1, 1) * idx[:, 2].max()), idx.select(1, 2))
    idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() * idx[:, 3].max() + idx.select(1, 1) * idx[:, 2].max() * idx[:, 3].max() + idx.select(1, 2) * idx[:, 3].max() + idx.select(1, 3)
    _unique, inverse, counts = torch.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)
    
    if _unique.shape[0] < indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
        features_new.index_add_(0, inverse.long(), features)
        features = features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        indices = indices[perm_].int()

        if not features_add is None:
            features_add_new = torch.zeros((_unique.shape[0],), device=features_add.device)
            features_add_new.index_add_(0, inverse.long(), features_add)
            features_add = features_add_new / counts
    return features, indices, features_add

def split_voxels(x, b, imps_3d, voxels_3d, kernel_offsets, mask_multi=True, topk=True, threshold=0.5, only_return_add=False):
    """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            x: [N, C], input sparse features
            b: int, batch size id
            imps_3d: [N, kernelsize**3], the prediced importance values
            voxels_3d: [N, 3], the 3d positions of voxel centers 
            kernel_offsets: [kernelsize**3, 3], the offset coords in an kernel
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
    """
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    mask_voxel = imps_3d[batch_index, -1].sigmoid()
    mask_kernel = imps_3d[batch_index, :-1].sigmoid()

    if mask_multi:
        features_ori *= mask_voxel.unsqueeze(-1)

    if topk:
        _, indices = mask_voxel.sort(descending=True)
        indices_fore = indices[:int(mask_voxel.shape[0]*threshold)]
        indices_back = indices[int(mask_voxel.shape[0]*threshold):]
    else:
        indices_fore = mask_voxel > threshold
        indices_back = mask_voxel <= threshold

    features_fore = features_ori[indices_fore]
    coords_fore = indices_ori[indices_fore]

    mask_kernel_fore = mask_kernel[indices_fore]
    mask_kernel_bool = mask_kernel_fore>=threshold
    voxel_kerels_imp = kernel_offsets.unsqueeze(0).repeat(mask_kernel_bool.shape[0],1, 1)
    mask_kernel_fore = mask_kernel[indices_fore][mask_kernel_bool]
    indices_fore_kernels = coords_fore[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
    indices_with_imp = indices_fore_kernels + voxel_kerels_imp
    selected_indices = indices_with_imp[mask_kernel_bool]
    spatial_indices = (selected_indices[:, 0] >0) * (selected_indices[:, 1] >0) * (selected_indices[:, 2] >0)  * \
                        (selected_indices[:, 0] < x.spatial_shape[0]) * (selected_indices[:, 1] < x.spatial_shape[1]) * (selected_indices[:, 2] < x.spatial_shape[2])
    selected_indices = selected_indices[spatial_indices]
    mask_kernel_fore = mask_kernel_fore[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_fore.device)*b, selected_indices], dim=1)

    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_fore.device)

    if only_return_add:
        return selected_features, selected_indices.int(), mask_kernel_fore

    features_fore_cat = torch.cat([features_fore, selected_features], dim=0)
    coords_fore = torch.cat([coords_fore, selected_indices], dim=0)
    mask_kernel_fore = torch.cat([torch.ones(features_fore.shape[0], device=features_fore.device), mask_kernel_fore], dim=0)

    features_fore, coords_fore, mask_kernel_fore = check_repeat(features_fore_cat, coords_fore, features_add=mask_kernel_fore)

    features_back = features_ori[indices_back]
    coords_back = indices_ori[indices_back]

    return features_fore, coords_fore, features_back, coords_back, mask_kernel_fore


# ---------------------------method 2 --------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class fusion(nn.Module):

    def __init__(self, dim, cluster_feature=2304):
        super().__init__()
        self.FusionModel = Frequency_attention(dim=dim, num_heads=2, mlp_ratio=4)
        self.cluster1 = nn.AdaptiveAvgPool1d(cluster_feature)
 
    def forward(self, x):
        x_copy = x
        #size = x.shape[1]
        #print(x.shape)
        x = self.cluster1(x.transpose(1,2).contiguous())
        #print("x1",x.shape)
        x = self.FusionModel(x.transpose(1,2))
        #print("x2:",x.shape)
        x = nn.functional.interpolate(x.transpose(1,2), size=x_copy.shape[1])
        #print("x3:",x.shape)
        return x.transpose(1,2) + x_copy
    
class Frequency_attention(nn.Module):

    def __init__(self, dim, num_heads=2, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        num_heads= num_heads # 4 for tiny, 6 for small and 12 for base
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim,  num_heads=num_heads, qkv_bias=True, qk_scale=False, attn_drop=drop, proj_drop=drop)
 
    def forward(self, x):
        # x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x