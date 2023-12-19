import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import torch

import utils

font = {"size": 10, "family": "Arial"}
plt.rc("font", **font)

def rms_normalise_audio(x):
    x = x - torch.mean(x)
    rms = torch.sqrt(torch.mean(torch.pow(x, 2)))
    return x / (10 * rms)


def plot_attn_audio(signal_ax, spec_ax, color_ax, audio, fs, attention, t0=0, t1=None):
    ATTENTION_FS = 20
    if t1 is None:
        t1 = audio.duration

    audio = audio[int(t0 * fs) : int(t1 * fs)]
    attention = attention[
        int(t0 * ATTENTION_FS) : int((t1 + 0.1) * ATTENTION_FS)
    ]  # Add a bit so attention not cutoff at very right

    spec, spec_fs = utils.spectrogram(audio, fs)

    # Plot stethoscope signal
    t_audio = np.arange(len(audio)) / fs
    signal_ax.plot(
        t_audio, rms_normalise_audio(torch.as_tensor(audio)), c="gray", label="Recording"
    )
    signal_ax.set_ylabel("Normalised amplitude")
    signal_ax.set_xlim(0, t1 - t0)
    signal_ax.set_ylim(-0.85, 0.85)
    signal_ax.text(0.1, 0.63, "(a)")

    # Plot attention weights on twin of top axis
    twin_x = signal_ax.twinx()
    t_attention = np.arange(len(attention)) / ATTENTION_FS
    twin_x.plot(t_attention, attention, c="k", lw=1.5, label="Attention weights")
    twin_x.set_ylim(0)
    twin_x.set_ylabel("Attention weight")

    # Plot spectrogram on bottom axis
    x = np.arange(spec.shape[1]) / spec_fs
    y = np.arange(spec.shape[0]) * (1 / 0.1)
    X, Y = np.meshgrid(x, y)
    Y = Y - 0.5
    mesh = spec_ax.pcolormesh(X, Y, spec, vmin=-12, vmax=6)
    c = plt.colorbar(mesh, cax=color_ax)
    c.set_label("Power (dB/Hz)")
    spec_ax.set_ylabel("Frequency (Hz)")
    spec_ax.set_xlabel("Time (seconds)")
    spec_ax.set_ylim(0, 500)
    spec_ax.set_yticks([0, 100, 200, 300, 400, 500])
    spec_ax.text(0.1, 450, "(b)", c="white")

    fig.align_ylabels((signal_ax, spec_ax))
    fig.legend(loc=(0.135, 0.89), ncol=2)
    plt.setp(signal_ax.get_xticklabels(), visible=False)

    return signal_ax


fs, series = scipy.io.wavfile.read("data/attention_example.wav")

fig = plt.figure(figsize=(6.2, 3.4), dpi=300)
gs = matplotlib.gridspec.GridSpec(
    2, 2, wspace=0.04, hspace=0.12, width_ratios=[20, 1], left=0.05, right=0.95, top=0.95
)
signal_ax = fig.add_subplot(gs[0])
spec_ax = fig.add_subplot(gs[2], sharex=signal_ax)
color_ax = fig.add_subplot(gs[3])

attn = np.loadtxt("data/attention_weights.csv", delimiter=",")

plot_attn_audio(signal_ax, spec_ax, color_ax, series, fs, attn, 7, 17)
plt.savefig("figures/attention.png", bbox_inches="tight")
