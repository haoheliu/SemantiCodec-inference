import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from semanticodec.modules.decoder.latent_diffusion.util import (
    exists,
    default,
    count_params,
    instantiate_from_config,
)
from semanticodec.modules.decoder.latent_diffusion.modules.ema import LitEma

from semanticodec.modules.decoder.latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)

from semanticodec.modules.decoder.latent_diffusion.models.ddim import DDIMSampler
from semanticodec.modules.decoder.latent_diffusion.util import disabled_train
from semanticodec.utils import PositionalEncoding


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        sampling_rate=None,
        timesteps=1000,
        beta_schedule="linear",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.state = None
        assert sampling_rate is not None
        self.validation_folder_name = "temp_name"
        self.clip_denoised = clip_denoised
        self.first_stage_key = first_stage_key
        self.sampling_rate = sampling_rate

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size
        self.v_posterior = v_posterior

        self.channels = channels
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            # print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        self.logvar = nn.Parameter(self.logvar, requires_grad=False)
        self.pos_embed = PositionalEncoding(seq_length=512, embedding_dim=192)

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1 - self.alphas_cumprod)
                )
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            # print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                # print(f"{context}: Restored training weights")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        shape = (batch_size, channels, self.latent_t_size, self.latent_f_size)
        channels = self.channels
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        scale_factor=1.0,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        *args,
        **kwargs,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std

        self.evaluation_params = evaluation_params
        assert self.num_timesteps_cond <= kwargs["timesteps"]

        conditioning_key = list(cond_stage_config.keys())

        self.conditioning_key = conditioning_key

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_models = nn.ModuleList([])
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.conditional_dry_run_finished = False
        self.restarted_from_ckpt = False

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1.0 / self.scale_factor * z
            decoding = self.first_stage_model.decode(z)
        return decoding

    def mel_spectrogram_to_waveform(self, mel):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        return waveform

    def encode_first_stage(self, x):
        with torch.no_grad():
            return self.first_stage_model.encode(x)

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        # print("Use ddim sampler")

        ddim_sampler = DDIMSampler(self, device = self.device)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            batch_size,
            shape,
            cond,
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            mask=mask,
            **kwargs,
        )
        return samples, intermediates

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_recon = self.model(x_noisy, t, cond_dict=cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def generate_sample(
        self,
        quanized_feature,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        unconditional_guidance_scale=1.0,
    ):
        batch_size = quanized_feature.shape[0]

        pe = self.pos_embed(quanized_feature)

        unconditional_conditioning = {}
        if unconditional_guidance_scale != 1.0:
            unconditional_quanized_feature = torch.cat(
                [
                    quanized_feature * 0.0,
                    pe.repeat(quanized_feature.size(0), 1, 1).to(
                        quanized_feature.device
                    ),
                ],
                dim=-1,
            )
            unconditional_conditioning = {
                "crossattn_audiomae_pooled": [
                    unconditional_quanized_feature,
                    torch.ones(
                        (
                            unconditional_quanized_feature.size(0),
                            unconditional_quanized_feature.size(1),
                        )
                    )
                    .to(unconditional_quanized_feature.device)
                    .float(),
                ]
            }

        quanized_feature = torch.cat(
            [
                quanized_feature,
                pe.repeat(quanized_feature.size(0), 1, 1).to(quanized_feature.device),
            ],
            dim=-1,
        )
        latent = {
            "crossattn_audiomae_pooled": [
                quanized_feature,
                torch.ones((quanized_feature.size(0), quanized_feature.size(1)))
                .to(quanized_feature.device)
                .float(),
            ]
        }

        samples, _ = self.sample_log(
            cond=latent,
            batch_size=batch_size,
            x_T=x_T,
            ddim=True,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )

        mel = self.decode_first_stage(samples)

        return self.mel_spectrogram_to_waveform(mel)


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, cond_dict: dict = {}):
        x = x.contiguous()
        t = t.contiguous()
        context_list, attn_mask_list = [], []
        context, attn_mask = cond_dict["crossattn_audiomae_pooled"]
        context_list.append(context)
        attn_mask_list.append(attn_mask)
        out = self.diffusion_model(
            x,
            t,
            context_list=context_list,
            y=None,
            context_attn_mask_list=attn_mask_list,
        )
        return out


def extract_encoder_state_dict(checkpoint_path):
    state_dict = torch.load(checkpoint_path)["state_dict"]
    new_state_dict = {}
    for key in state_dict.keys():
        if "cond_stage_models.0" in key:
            if "pos_embed.pe" in key:
                continue
            new_key_name = key.replace("cond_stage_models.0.", "")
            new_state_dict[new_key_name] = state_dict[key]
    return new_state_dict


def overlap_add_waveform(windowed_waveforms, overlap_duration=0.64):
    """
    Concatenates a series of windowed waveforms with overlap, applying fade-in and fade-out effects to the overlaps.

    Parameters:
    - windowed_waveforms: a list of numpy arrays with shape (1, 1, samples_per_waveform)

    Returns:
    - A single waveform numpy array resulting from the overlap-add process.
    """
    # Assuming a sampling rate of 16000 Hz and 0.64 seconds overlap
    if overlap_duration < 1e-4:
        return np.concatenate(windowed_waveforms, axis=-1)

    sampling_rate = 16000
    overlap_samples = int(overlap_duration * sampling_rate)

    # Initialize the output waveform
    output_waveform = np.array([]).reshape(1, 1, -1)

    for i, waveform in enumerate(windowed_waveforms):
        # If not the first waveform, apply fade-in at the beginning
        if i > 0:
            fade_in = np.linspace(0, 1, overlap_samples).reshape(1, 1, -1)
            waveform[:, :, :overlap_samples] *= fade_in

        # If output waveform already has content, apply fade-out to its last overlap and add the overlapping parts
        if output_waveform.size > 0:
            fade_out = np.linspace(1, 0, overlap_samples).reshape(1, 1, -1)
            # Apply fade-out to the end of the output waveform
            output_waveform[:, :, -overlap_samples:] *= fade_out
            # Add the faded-in start of the current waveform to the faded-out end of the output waveform
            output_waveform[:, :, -overlap_samples:] += waveform[:, :, :overlap_samples]

        # Concatenate the current waveform (minus the initial overlap if not the first) to the output
        if output_waveform.size == 0:
            output_waveform = waveform
        else:
            output_waveform = np.concatenate(
                (output_waveform, waveform[:, :, overlap_samples:]), axis=2
            )

    return output_waveform
