"""
Reference Repo: https://github.com/facebookresearch/AudioMAE
"""

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import semanticodec.modules.audiomae.models_mae as models_mae

# model = mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024, 128))


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Vanilla_AudioMAE(nn.Module):
    """Audio Masked Autoencoder (MAE) pre-trained on AudioSet (for AudioLDM)"""

    def __init__(
        self,
    ):
        super().__init__()
        model = models_mae.__dict__["mae_vit_base_patch16"](
            in_chans=1, audio_exp=True, img_size=(1024, 128)
        )

        # checkpoint_path = "/mnt/bn/lqhaoheliu/exps/checkpoints/audiomae/pretrained.pth"
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # model.load_state_dict(checkpoint["model"], strict=False)

        # Skip the missing keys of decoder modules (not required)
        # print(f'Load AudioMAE from {checkpoint_path} / message: {msg}')

        self.model = model.eval()

    def forward(self, x, mask_ratio=0.0, no_mask=False, no_average=False):
        """
        x: mel fbank [Batch, 1, 1024 (T), 128 (F)]
        mask_ratio: 'masking ratio (percentage of removed patches).'
        """
        with torch.no_grad():
            # embed: [B, 513, 768] for mask_ratio=0.0
            if no_mask:
                if no_average:
                    raise RuntimeError("This function is deprecated")
                    embed = self.model.forward_encoder_no_random_mask_no_average(
                        x
                    )  # mask_ratio
                else:
                    embed = self.model.forward_encoder_no_mask(x)  # mask_ratio
            else:
                raise RuntimeError("This function is deprecated")
                embed, _, _, _ = self.model.forward_encoder(x, mask_ratio=mask_ratio)
        return embed


if __name__ == "__main__":
    model = Vanilla_AudioMAE().cuda()
    input = torch.randn(4, 1, 1024, 128).cuda()
    print("The first run")
    embed = model(input, mask_ratio=0.0, no_mask=True)
    print(embed)
    print("The second run")
    embed = model(input, mask_ratio=0.0)
    print(embed)
