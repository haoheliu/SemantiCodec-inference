import torch
import math
import torch.nn as nn

import torchaudio


def concat_1x2(tensor):
    batchsize, width, height, channels = tensor.shape
    # Check if height is divisible by 2 for concatenation
    if height % 2 != 0:
        raise ValueError("Height must be divisible by 2 for 1x2 concatenation.")
    # Reshape to group 1x2 blocks
    tensor_reshaped = tensor.view(batchsize, width, height // 2, 2, channels)
    # Permute to move the 1x2 blocks next to the channel dimension
    tensor_permuted = tensor_reshaped.permute(0, 1, 2, 3, 4)
    # Concatenate the 1x2 blocks along the channel dimension
    tensor_concat = tensor_permuted.reshape(batchsize, width, height // 2, channels * 2)
    return tensor_concat


def concat_2x2(tensor):
    batchsize, width, height, channels = tensor.shape
    # Reshape to group 2x2 blocks
    tensor_reshaped = tensor.view(batchsize, width // 2, 2, height // 2, 2, channels)
    # Permute to move the 2x2 blocks next to the channel dimension
    tensor_permuted = tensor_reshaped.permute(0, 1, 3, 2, 4, 5)
    # Concatenate the 2x2 blocks along the channel dimension
    tensor_concat = tensor_permuted.reshape(
        batchsize, width // 2, height // 2, channels * 4
    )
    return tensor_concat


def extract_kaldi_fbank_feature(waveform, sampling_rate, target_length=1024):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    sampling_rate = sampling_rate

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = target_length

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]


class PositionalEncoding:
    def __init__(self, seq_length=512, embedding_dim=192):
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # Initialize positional embeddings
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(seq_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a 'batch' dimension with 'unsqueeze'
        self.pe = pe.unsqueeze(0)

    def __call__(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, embedding_dim]
        """
        # return positional embeddings
        return self.pe[:, : x.size(1)]
