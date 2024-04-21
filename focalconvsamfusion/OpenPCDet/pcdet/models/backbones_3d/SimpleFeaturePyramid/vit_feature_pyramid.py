import math
import torch
import torch.nn as nn
from pcdet.models.backbones_3d.SimpleFeaturePyramid.basic_utils import Conv2d, get_norm
import torch.nn.functional as F




class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_feature='last_feat',
        out_channels = 256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=None,
        dim = 256,
        norm="LN",
        square_pad=0,
        input_shapes=64,
        fuse_type="avg"
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        dim = dim
        self.stages = []
        input_shapes = input_shapes
        strides = [int(input_shapes / scale) for scale in scale_factors]
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        output_convs = nn.ModuleList()
        for idx, scale_factor in enumerate(scale_factors):
            if idx == len(scale_factors) -1:
                continue
            output_norm = get_norm(norm, out_channels)

            # lateral_conv = Conv2d(
            #     in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            # )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            
            self.add_module("fpn_output{}".format(stage), output_conv)
            output_convs.append(output_conv)
        self.output_convs = output_convs
            
            
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self.fuse_type = fuse_type

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, bottom_up_features):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        features = bottom_up_features
        results = []
        # get ms feat
        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
            
        # fuse ms feat
        prev_feature = results[-1] # 256, 256
        for idx, output_conv in enumerate(self.output_convs):
            feature = results[len(self.output_convs) - 1 - idx]
            top_down_feature = F.interpolate(prev_feature, scale_factor=2.0, mode="nearest")
            prev_feature = feature + top_down_feature
            if self.fuse_type == "avg":
                prev_feature /= 2
            results[len(self.output_convs)-1-idx] =  output_conv(prev_feature)
            
            
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
    
    
if __name__ == '__main__':
    model = SimpleFeaturePyramid(scale_factors=[4.0, 2.0, 1.0, 0.5])
    x = torch.randn([1, 256, 64, 64])
    out = model(x)
    # print(out['p4'].shape)