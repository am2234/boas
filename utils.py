import numpy as np
import sklearn.metrics
import torch
import math

def calc_binary_metrics(pred, target):
    mat = sklearn.metrics.confusion_matrix(target, pred)
    tn, fp, fn, tp = mat.ravel()
    se = (tp) / (tp + fn)
    sp = (tn) / (tn + fp)
    ppv = tp / (tp + fp)
    return se, sp, ppv


def prepare_roc_plot(axes, grid=True):
    axes.plot([0, 100], [0, 100], "--", label="Random classifier", c="lightgray")
    axes.set_xlabel("100 - Specificity (%)")
    axes.set_ylabel("Sensitivity (%)")
    axes.set_aspect("equal")
    axes.set_xlim((-0.025, 100))
    axes.set_ylim((0, 100.01))
    if grid:
        axes.grid(alpha=0.2)


def plot_averaged_roc(
    axes,
    pairs,
    std_dev=2,
    alpha_runs=0.05,
    alpha_fill=0.15,
    op_point=False,
    fpr_range=None,
    tpr_range=None,
    op_decimals=1,
):
    col = "k"
    mean_fpr = np.linspace(0, 1, 201)
    mean_fpr = np.insert(mean_fpr, 0, 0)
    aucs = []
    tprs = []
    for i, (posterior, target) in enumerate(pairs):
        label_add = "" if i == 0 else "_"
        fpr, tpr, _ = sklearn.metrics.roc_curve(target, posterior)
        axes.plot(
            100 * fpr,
            100 * tpr,
            label=label_add + "Individual runs",
            c="gray",
            alpha=alpha_runs,
            lw=1,
        )
        this_auc = sklearn.metrics.auc(fpr, tpr)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(this_auc)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_dev * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_dev * std_tpr, 0)

    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    std_auc = std_dev * np.std(aucs)

    axes.plot(
        100 * mean_fpr,
        100 * mean_tpr,
        color=col,
        label="Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        alpha=1,
    )
    axes.fill_between(
        100 * mean_fpr,
        100 * tprs_lower,
        100 * tprs_upper,
        color=col,
        alpha=alpha_fill,
        label=f"$\pm$ {std_dev} std. dev.",
    )

    if op_point:
        search_range = np.ones(len(mean_fpr), dtype=bool)
        if fpr_range is not None:
            search_range = search_range & (mean_fpr >= fpr_range[0]) & (mean_fpr <= fpr_range[1])
        if tpr_range is not None:
            search_range = search_range & (mean_tpr >= tpr_range[0]) & (mean_tpr <= tpr_range[1])
        if not search_range.any():
            raise ValueError("No valid points on curve in search range")

        best = np.argmax((mean_tpr - mean_fpr) * search_range)
        op_se = mean_tpr[best]
        op_sp = 1 - mean_fpr[best]

        op_se_std = std_tpr[best]

        axes.plot(
            100 * mean_fpr[best],
            100 * op_se,
            "o",
            c=col,
            label=f"OP (Se = {op_se*100:.{op_decimals}f}%, Sp = {op_sp*100:.{op_decimals}f}%)",
        )
        print(1 - mean_fpr[best], mean_tpr[best], std_tpr[best])
        print(
            f" Se = {op_se*100:.1f}% (95% CI {100*(op_se-2*op_se_std):.1f}%-{100*(op_se+2*op_se_std):.1f}%)"
        )

    return aucs, mean_fpr, mean_tpr, tprs_lower, tprs_upper


def spectrogram(x, fs):
    WIN_LENGTH = 0.1  # millseconds
    WIN_OVERLAP = 0.5  # fraction
    low_freq = 0  # Hz
    high_freq = 500  # Hz

    win_step = WIN_OVERLAP * WIN_LENGTH
    win_samples = int(WIN_LENGTH * fs)

    x = torch.from_numpy(x.copy())

    # Normalise to zero-mean peak unit value
    x = x - torch.mean(x)
    x = x / torch.max(torch.abs(x))

    # Compute log-spectrogram
    spec = torch.stft(
        x,
        n_fft=win_samples,
        hop_length=int(win_step * fs),
        win_length=win_samples,
        window=torch.hann_window(win_samples),
        return_complex=False,
    )
    spec = spec.pow(2).sum(axis=-1)
    spec = torch.log(spec)

    # Select spectrogram in approximately 20 - 200 Hz range
    freq_resolution = 1 / WIN_LENGTH
    low_bin = math.floor(low_freq / freq_resolution)
    high_bin = math.ceil(high_freq / freq_resolution)
    spec = spec[low_bin : high_bin + 1, :]

    return spec.float(), int(1 / win_step)