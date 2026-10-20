from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_convergence(histories: dict, out_path: str | None = None,
                     ylabel: str = "f(x)", logy: bool = True,
                     title: str | None = None):
    """histories: {label: History}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, h in histories.items():
        if logy:
            ax.semilogy(h.f, label=label)
        else:
            ax.plot(h.f, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_grad_norm(histories: dict, out_path: str | None = None,
                   tol: float | None = None, title: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, h in histories.items():
        ax.semilogy(h.grad_norm, label=label)
    if tol is not None:
        ax.axhline(tol, color="black", linestyle="--", alpha=0.5,
                   label=f"tol={tol:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||grad||")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_trajectories_2d(oracle, histories: dict, out_path: str | None = None,
                         xlim=None, ylim=None, title: str | None = None):
    """For 2D problems: contour plot + trajectories."""
    if oracle.dim != 2:
        raise ValueError("plot_trajectories_2d requires dim=2.")

    if xlim is None or ylim is None:
        all_x = np.concatenate([np.array(h.x) for h in histories.values()])
        xlim = (all_x[:, 0].min() - 1, all_x[:, 0].max() + 1)
        ylim = (all_x[:, 1].min() - 1, all_x[:, 1].max() + 1)

    xs = np.linspace(*xlim, 200)
    ys = np.linspace(*ylim, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = oracle.f(np.array([XX[i, j], YY[i, j]]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(XX, YY, ZZ, levels=30, cmap="viridis", alpha=0.7)
    for label, h in histories.items():
        traj = np.array(h.x)
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, label=label)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_heatmap(values: np.ndarray, row_labels, col_labels,
                 out_path: str | None = None, cmap: str = "viridis",
                 title: str | None = None, fmt: str = ".2f"):
    """values: (n_rows, n_cols) array."""
    fig, ax = plt.subplots(figsize=(1.2 * len(col_labels) + 2,
                                    0.6 * len(row_labels) + 2))
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, format(values[i, j], fmt),
                    ha="center", va="center", color="white")
    plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    if out_path:
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig, ax