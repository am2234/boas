
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


def plot_signal_and_spec(filename, t0, t1):
    fs, series = scipy.io.wavfile.read(filename)
    norm_audio_series = rms_normalise_audio(torch.as_tensor(series)).numpy()
    norm_audio_series = norm_audio_series[int(t0 * fs) : int(t1 * fs)]

    spec, spec_fs = utils.spectrogram(series, fs)
    spec = spec[:, int(spec_fs * t0) : int(spec_fs * t1) + 2]

    fig = plt.figure(figsize=(6.2, 3.4), dpi=300)
    gs = plt.GridSpec(
        2, 2, wspace=0.04, hspace=0.08, width_ratios=[20, 1], left=0.05, right=0.95, top=0.95
    )
    signal_ax = fig.add_subplot(gs[0])
    spec_ax = fig.add_subplot(gs[2], sharex=signal_ax)
    color_ax = fig.add_subplot(gs[3])

    t_series = np.arange(len(norm_audio_series)) / fs
    signal_ax.plot(t_series, norm_audio_series, c=(0.3,) * 3)
    signal_ax.get_xaxis().set_visible(False)
    signal_ax.set_ylabel("Normalised amplitude")
    # signal_ax.set_yticks([])
    signal_ax.set_ylim(-0.42, 0.42)

    spec_ax.set_ylabel("Frequency (Hz)")
    spec_ax.set_xlabel("Time (seconds)")
    spec_ax.set_ylim(0, 500)
    spec_ax.set_yticks([0,100,200,300,400,500])
    x = np.arange(spec.shape[1]) / spec_fs
    y = np.arange(spec.shape[0]) * (1 / 0.1)
    X, Y = np.meshgrid(x, y)
    Y = Y - 0.5

    mesh = spec_ax.pcolormesh(X, Y, spec, vmin=-12, vmax=6)
    c = plt.colorbar(mesh, cax=color_ax)
    c.set_label("Power (dB/Hz)")
    spec_ax.set_xlim(0, t1 - t0)

    signal_ax.text(0.1, 0.33, "(a)")
    spec_ax.text(0.1, 450, "(b)", c="white")

    fig.align_ylabels((signal_ax, spec_ax))


# Moderate (constant) stertor
plot_signal_and_spec("data/moderate_stertor.wav", 17, 26)
plt.savefig("figures/stertor_spec.png", bbox_inches="tight")

# Healthy
plot_signal_and_spec("data/healthy.wav", 5, 15)
plt.savefig("figures/healthy_spec.png", bbox_inches="tight")
