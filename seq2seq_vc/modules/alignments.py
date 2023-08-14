# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba
from numba import jit


class AlignmentModule(nn.Module):
    """Alignment Learning Framework proposed for parallel TTS models in:

    https://arxiv.org/abs/2108.10447

    """

    def __init__(self, adim, odim):
        super().__init__()
        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

        self.f_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.f_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.f_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, text, feats, x_masks=None):
        """Calculate alignment loss.

        Args:
            text (Tensor): Batched text embedding (B, T_text, adim).
            feats (Tensor): Batched acoustic feature (B, T_feats, odim).
            x_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Log probability of attention matrix (B, T_feats, T_text).

        """
        text = text.transpose(1, 2)
        text = F.relu(self.t_conv1(text))
        text = self.t_conv2(text)
        text = text.transpose(1, 2)

        feats = feats.transpose(1, 2)
        feats = F.relu(self.f_conv1(feats))
        feats = F.relu(self.f_conv2(feats))
        feats = self.f_conv3(feats)
        feats = feats.transpose(1, 2)

        dist = feats.unsqueeze(2) - text.unsqueeze(1)
        dist = torch.norm(dist, p=2, dim=3)
        score = -dist

        if x_masks is not None:
            x_masks = x_masks.unsqueeze(-2)
            score = score.masked_fill(x_masks, -np.inf)

        log_p_attn = F.log_softmax(score, dim=-1)
        return log_p_attn


@jit(nopython=True)
def _monotonic_alignment_search(log_p_attn):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A


@jit(nopython=True)
def _monotonic_alignment_search_v2(log_p_attn):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            Q[i, j] = max(
                Q[i - 1, j - 1] * 2 + log_prob[i, j],
                Q[i, j - 1] + log_prob[i, j],
                Q[i - 1, j] + log_prob[i, j],
            )

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-2, A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 2
        i_b = A[j + 1] - 1
        i_c = A[j + 1]
        if i_c == 0:
            argmax_i = 0
        else:
            if i_b == 0:  # just compare b and c
                if Q[i_b, j] >= Q[i_c, j]:
                    argmax_i = i_b
                else:
                    argmax_i = i_c
            else:  # compare a and b and c

                if Q[i_a, j] >= Q[i_b, j] and Q[i_a, j] >= Q[i_c, j]:
                    argmax_i = i_a
                elif Q[i_b, j] > Q[i_a, j] and Q[i_b, j] >= Q[i_c, j]:
                    argmax_i = i_b
                else:
                    argmax_i = i_c
        A[j] = argmax_i
    return A


@jit(nopython=True)
def _monotonic_alignment_search_k(log_p_attn, k):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            candidates = [Q[i - _k, j - 1] for _k in range(k) if i - _k >= 0]
            Q[i, j] = max(candidates) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-2, A[j+1]-1, A[j+1]}
        candidates = [A[j + 1]]
        for _k in range(1, k):
            candidate = A[j + 1] - _k
            if candidate < 0:
                break
            else:
                candidates.append(candidate)

        # argmax
        argmax_i = candidates[0]
        current_max = Q[argmax_i, j]
        for candidate in candidates[1:]:
            if Q[candidate, j] >= current_max:  # ">=": aggressively move backward
                argmax_i = candidate
                current_max = Q[candidate, j]
        A[j] = argmax_i
    return A


@jit(nopython=True)
def _monotonic_alignment_search_v4_k(log_p_attn, k):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_mel):
        for i in range(1, min(j + 1, T_inp)):
            candidates = [Q[i - _k, j - 1] for _k in range(k) if i - _k >= 0]
            Q[i, j] = max(candidates) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-2, A[j+1]-1, A[j+1]}
        candidates = [A[j + 1]]
        for _k in range(1, k):
            candidate = A[j + 1] - _k
            if candidate < 0:
                break
            else:
                candidates.append(candidate)

        # argmax
        argmax_i = candidates[0]
        current_max = Q[argmax_i, j]
        for candidate in candidates[1:]:
            if Q[candidate, j] > current_max:  # ">=": aggressively move backward
                argmax_i = candidate
                current_max = Q[candidate, j]
        A[j] = argmax_i
    return A


@jit((numba.float64[:, :], numba.int8, numba.boolean), nopython=True)
def _monotonic_alignment_search_v5(log_p_attn, k, transpose):
    # https://arxiv.org/abs/2005.11129
    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1, 0)  # -> (T_inp,T_mel)

    # the usual case
    if not transpose:
        # 1.  Q <- init first row for all j
        for j in range(T_mel):
            Q[0, j] = log_prob[0, : j + 1].sum()

        # 2.
        for j in range(1, T_mel):
            for i in range(1, min(j + 1, T_inp)):
                candidates = [Q[i - _k, j - 1] for _k in range(k) if i - _k >= 0]
                Q[i, j] = max(candidates) + log_prob[i, j]
    else:
        # 1.  Q <- init first row for all j
        for i in range(T_inp):
            Q[i, 0] = log_prob[: i + 1, 0].sum()

        # 2.
        for i in range(1, T_inp):
            for j in range(1, min(i + 1, T_mel)):
                # candidates = [Q[i - _k, j - 1] for _k in range(k) if i - _k >= 0]
                candidates = [Q[i - 1, j - _k] for _k in range(k) if j - _k >= 0]
                Q[i, j] = max(candidates) + log_prob[i, j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp - 1)
    for j in range(T_mel - 2, -1, -1):  # T_mel-2, ..., 0
        # 'i' in {A[j+1]-2, A[j+1]-1, A[j+1]}
        candidates = [A[j + 1]]
        for _k in range(1, k):
            candidate = A[j + 1] - _k
            if candidate < 0:
                break
            else:
                candidates.append(candidate)

        # argmax
        argmax_i = candidates[0]
        current_max = Q[argmax_i, j]
        for candidate in candidates[1:]:
            if Q[candidate, j] > current_max:  # ">=": aggressively move backward
                argmax_i = candidate
                current_max = Q[candidate, j]
        A[j] = argmax_i
    return A


def viterbi_decode(log_p_attn, text_lengths, feats_lengths, k=None):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search(cur_log_p_attn.detach().cpu().numpy())
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


def viterbi_decode_v2(log_p_attn, text_lengths, feats_lengths, k=None):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search_v2(cur_log_p_attn.detach().cpu().numpy())
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


def viterbi_decode_k(log_p_attn, text_lengths, feats_lengths, k=2):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).
        k (int): number of frames to backtrack

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    assert k >= 2
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search_k(
            cur_log_p_attn.detach().cpu().numpy(), k
        )
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss


def viterbi_decode_v4_k(log_p_attn, text_lengths, feats_lengths, k=2):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).
        k (int): number of frames to backtrack

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    assert k >= 2
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search_v4_k(
            cur_log_p_attn.detach().cpu().numpy(), k
        )
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss

# v5 changes the direction according to the length of the input and output
# but k is not implemented yet
def viterbi_decode_v5(log_p_attn, text_lengths, feats_lengths, k=2):
    """Extract duration from an attention probability matrix

    Args:
        log_p_attn (Tensor): Batched log probability of attention
            matrix (B, T_feats, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feats_legnths (Tensor): Feature length tensor (B,).
        k (int): number of frames to backtrack

    Returns:
        Tensor: Batched token duration extracted from `log_p_attn` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    assert k >= 2
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    bin_loss = 0
    ds = torch.zeros((B, T_text), device=device)
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, : feats_lengths[b], : text_lengths[b]]
        transpose = bool(feats_lengths[b] >= text_lengths[b])
        viterbi = _monotonic_alignment_search_v5(
            cur_log_p_attn.detach().cpu().numpy().astype(np.float64),
            np.int8(k),
            transpose,
        )
        _ds = np.bincount(viterbi)
        ds[b, : len(_ds)] = torch.from_numpy(_ds).to(device)

        t_idx = torch.arange(feats_lengths[b])
        bin_loss = bin_loss - cur_log_p_attn[t_idx, viterbi].mean()
    bin_loss = bin_loss / B
    return ds, bin_loss

@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """Average frame-level features into token-level according to durations

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text).

    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg
