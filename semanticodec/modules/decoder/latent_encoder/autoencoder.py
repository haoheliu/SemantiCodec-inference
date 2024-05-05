import torch
import os

import torch.nn.functional as F
from semanticodec.modules.decoder.latent_diffusion.modules.ema import *

from semanticodec.modules.decoder.latent_diffusion.modules.diffusionmodules.model import Encoder, Decoder
from semanticodec.modules.decoder.latent_diffusion.modules.distributions.distributions import (
    DiagonalGaussianDistribution,
)

import soundfile as sf

from semanticodec.modules.decoder.utilities.model import get_vocoder
from semanticodec.modules.decoder.utilities.tools import synth_one_sample

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()
        self.automatic_optimization=False
        assert "mel_bins" in ddconfig.keys(), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, "cpu", num_mel)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.learning_rate = float(base_learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None

    def get_log_dir(self):
        return os.path.join(self.logger_save_dir, self.logger_exp_group_name, self.logger_exp_name)

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_name = exp_name
        self.logger_exp_group_name = exp_group_name

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # x = self.time_shuffle_operation(x)
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # bs, ch, shuffled_timesteps, fbins = dec.size()
        # dec = self.time_unshuffle_operation(dec, bs, int(ch*shuffled_timesteps), fbins)
        dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):
        from utilities.model import vocoder_infer

        if self.image_key == "fbank":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = vocoder_infer(dec, self.vocoder)
        elif self.image_key == "stft":
            dec = dec.squeeze(1).permute(0, 2, 1)
            wav_reconstruction = self.wave_decoder(dec)
        return wav_reconstruction

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if self.flag_first_run:
            print("Latent size: ", z.size())
            self.flag_first_run = False

        dec = self.decode(z)

        return dec, posterior

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)

    def save_wave(self, batch_wav, fname, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for wav, name in zip(batch_wav, fname):
            name = os.path.basename(name)

            sf.write(os.path.join(save_dir, name), wav, samplerate=self.sampling_rate)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, train=True, only_inputs=False, waveform=None, **kwargs):
        log = dict()
        x = batch.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(posterior.sample())
            log["reconstructions"] = xrec

        log["inputs"] = x
        wavs = self._log_img(log, train=train, index=0, waveform=waveform)
        return wavs

    def tensor2numpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

