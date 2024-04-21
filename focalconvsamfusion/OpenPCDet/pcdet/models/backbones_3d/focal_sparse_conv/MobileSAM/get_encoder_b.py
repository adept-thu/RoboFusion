import torch
from mobile_sam.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, TinyViT, Package_Tiny_ViT
from functools import partial

sam_checkpoint = "./weights/mobile_sam.pt"
model_type = "vit_t"
device = "cuda"
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size


sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

package_image_encoder = Package_Tiny_ViT(TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),)

premodel = torch.load(sam_checkpoint)
sam.load_state_dict(premodel)
# print(sam)

model_image_encoder_dict = package_image_encoder.state_dict()
state_dict = {k:v for k,v in premodel.items() if k in model_image_encoder_dict.keys()}

model_image_encoder_dict.update(state_dict)
package_image_encoder.load_state_dict(model_image_encoder_dict)
# print(package_image_encoder)
torch.save(package_image_encoder.state_dict(), '/home/zhanggl/sda/szy/focalsconv-mm/OpenPCDet/checkpoints/mobile_image_encoder.pt')
print("end")