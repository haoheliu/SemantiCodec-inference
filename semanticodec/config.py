
def get_config(token_rate=100, vocab_size=None, checkpoint_path=None):
    assert vocab_size in [4096, 8192, 16384, 32768], "vocab_size must be 4096, 8192, 16384 or 32768"
    assert token_rate in [25, 50, 100], "token_rate must be 25, 50 or 100"

    semantic_codebook = {
        25: {
            4096: "pretrained/codebook_idx/combine_128_audioset_dominate/codebook_2048_0.npy",
            8192: "pretrained/codebook_idx/combine_128_audioset_dominate/codebook_4096_0.npy",
            16384: "pretrained/codebook_idx/combine_128_audioset_dominate/codebook_8192_0.npy",
            32768: "pretrained/codebook_idx/combine_128_audioset_dominate/codebook_16384_0.npy",
        },
        50: {
            4096: "pretrained/codebook_idx/combine_256_audioset_dominate/codebook_2048_0.npy",
            8192: "pretrained/codebook_idx/combine_256_audioset_dominate/codebook_4096_0.npy",
            16384: "pretrained/codebook_idx/combine_256_audioset_dominate/codebook_8192_0.npy",
            32768: "pretrained/codebook_idx/combine_256_audioset_dominate/codebook_16384_0.npy",
        },
        100: {
            4096: "pretrained/codebook_idx/combine_512_audioset_dominate/codebook_2048_0.npy",
            8192: "pretrained/codebook_idx/combine_512_audioset_dominate/codebook_4096_0.npy",
            16384: "pretrained/codebook_idx/combine_512_audioset_dominate/codebook_8192_0.npy",
            32768: "pretrained/codebook_idx/combine_512_audioset_dominate/codebook_16384_0.npy",
        },
    }

    basic_config = {
    "model": {
        "params": {
        "latent_t_size": 256, 
        "scale_by_std": True, 
        "sampling_rate": 16000, 
        "first_stage_config": {
            "params": {
            "monitor": "val/rec_loss", 
            "image_key": "fbank", 
            "embed_dim": 8, 
            "batchsize": 16, 
            "reload_from_ckpt": "/mnt/bn/lqhaoheliu/exps/checkpoints/audioldm/vae_32k/2023_06_22_vae_16k_64_4/last.ckpt", 
            "subband": 1, 
            "time_shuffle": 1, 
            "sampling_rate": 16000, 
            "ddconfig": {
                "ch": 128, 
                "double_z": True, 
                "out_ch": 1, 
                "attn_resolutions": [], 
                "dropout": 0.0, 
                "mel_bins": 64, 
                "ch_mult": [
                1, 
                2, 
                4
                ], 
                "num_res_blocks": 2, 
                "z_channels": 8, 
                "downsample_time": False, 
                "in_channels": 1, 
                "resolution": 256
            }, 
            "lossconfig": {
                "params": {
                "disc_start": 50001, 
                "kl_weight": 1000.0, 
                "disc_in_channels": 1, 
                "disc_weight": 0.5
                }, 
                "target": "semanticodec.modules.decoder.latent_diffusion.modules.losses.LPIPSWithDiscriminator"
            }
            }, 
            "target": "semanticodec.modules.decoder.latent_encoder.autoencoder.AutoencoderKL", 
            "base_learning_rate": 8e-06
        }, 
        "unet_config": {
            "params": {
            "channel_mult": [
                1, 
                2, 
                3, 
                5
            ], 
            "out_channels": 8, 
            "attention_resolutions": [
                8, 
                4, 
                2
            ], 
            "context_dim": [
                1728
            ], 
            "num_res_blocks": 2, 
            "in_channels": 8, 
            "image_size": 64, 
            "transformer_depth": 1, 
            "use_spatial_transformer": True, 
            "model_channels": 64, 
            "num_head_channels": 32
            }, 
            "target": "semanticodec.modules.decoder.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel"
        }, 
        "base_learning_rate": 0.0001, 
        "channels": 8, 
        "linear_start": 0.0015, 
        "first_stage_key": "fbank", 
        "parameterization": "v", 
        "cond_stage_config": {
            "crossattn_audiomae_pooled": {
            "cond_stage_key": "ta_kaldi_fbank", 
            "params": {
                "use_oracle": False, 
                "lstm_bidirectional": True, 
                "feature_dimension": 768, 
                "codebook_size": 8192, 
                "residual_encoder": "lstm", 
                "rvq_layers": 0, 
                "lstm_layer": 4
            }, 
            "target": "semanticodec.modules.encoder.encoder.AudioMAEConditionQuantResEncoder", 
            "conditioning_key": "crossattn"
            }
        }, 
        "num_timesteps_cond": 1, 
        "timesteps": 1000, 
        "latent_f_size": 16, 
        "linear_end": 0.0195
        }, 
        "target": "semanticodec.modules.decoder.latent_diffusion.models.ddpm.LatentDiffusion"
    }
    }

    if token_rate == 50:
        # modify context_dim
        basic_config["model"]["params"]["unet_config"]["params"]["context_dim"] = [3264]
        # modify cond_stage_config
        basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["lstm_layer"] = 3
        basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["feature_dimension"] = 768 * 2
    elif token_rate == 25:
        # modify context_dim
        basic_config["model"]["params"]["unet_config"]["params"]["context_dim"] = [6336]
        # modify cond_stage_config
        basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["lstm_layer"] = 2
        basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["feature_dimension"] = 768 * 4
    elif token_rate == 100:
        pass
    else:
        raise ValueError("token_rate must be 50, 25 or 100")

    if checkpoint_path is None:
        checkpoint_path = "pretrained/semanticodec_tokenrate_%s" % token_rate
    else:
        print("Using custom checkpoint path: %s" % checkpoint_path)

    feature_dim = basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["feature_dimension"]
    lstm_layers = basic_config["model"]["params"]["cond_stage_config"]["crossattn_audiomae_pooled"]["params"]["lstm_layer"]
    return basic_config, checkpoint_path, feature_dim, lstm_layers, semantic_codebook[token_rate][vocab_size]