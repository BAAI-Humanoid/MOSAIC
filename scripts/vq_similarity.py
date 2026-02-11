#!/usr/bin/env python3
"""
Load a trained ActorCriticVQ checkpoint, feed motion features into the VQEncoder,
and visualize similarity/distance between latents before and after VQ quantization.

Supported motion inputs:
- npz_command: read joint_pos/joint_vel from .npz and concatenate into [T, 2*DoF]
- npz_command_anchor_identity: like npz_command, plus anchor pos/orientation terms (offline approximation)
- npz_key: read a specified key from .npz (must be [T, D])
- npy: read a .npy array (must be [T, D])
- csv: read a .csv (must be [T, D], comma-separated)

Examples:
  python scripts/vq_similarity.py \\
    --checkpoint outputs/2025-12-17/12-10-42/checkpoints/model_10000.pt \\
    --motion motions/PHUMA_G1_NPZ/aist/xxx.npz \\
    --motion_mode npz_command \\
    --out_dir /tmp/vq_vis

If your motion is already the encoder input feature matrix:
  python scripts/vq_similarity.py \\
    --checkpoint .../model.pt \\
    --motion /path/to/features.npy \\
    --motion_mode npy \\
    --out_dir /tmp/vq_vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import torch


def _add_repo_to_syspath() -> None:
    """Allow running via `python scripts/...py` without installing the package."""
    repo_root = Path(__file__).resolve().parents[1]
    rsl_rl_src = repo_root / "source" / "rsl_rl"
    if str(rsl_rl_src) not in sys.path:
        sys.path.insert(0, str(rsl_rl_src))


def _load_checkpoint_state_dict(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        # Handle DDP/DP checkpoints with "module.xxx" prefixes.
        if isinstance(sd, dict) and sd and all(isinstance(k, str) for k in sd.keys()):
            if any(k.startswith("module.") for k in sd.keys()):
                return {k[len("module.") :]: v for k, v in sd.items() if isinstance(k, str)}
        return sd
    if isinstance(ckpt, dict):
        # Handle checkpoints saved as torch.save(policy.state_dict()).
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            if any(k.startswith("module.") for k in ckpt.keys()):
                return {k[len("module.") :]: v for k, v in ckpt.items() if isinstance(k, str)}
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: type={type(ckpt)}, file={checkpoint_path}")


def _extract_sub_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    if not out:
        # Provide a more helpful error message.
        sample_keys = list(state_dict.keys())[:30]
        raise KeyError(
            f"Cannot find parameters with prefix `{prefix}` in the checkpoint. "
            f"Your checkpoint may not be from ActorCriticVQ. Example keys: {sample_keys}"
        )
    return out


def _infer_mlp_dims_from_state_dict(encoder_sd: Dict[str, torch.Tensor]) -> Tuple[int, List[int], int]:
    """
    Infer MLP(input_dim, hidden_dims, output_dim) from VQEncoder state_dict.
    Assumes naming from rsl_rl.networks.encoder.MLP: mlp.net.{idx}.weight / bias
    """
    weights: List[Tuple[int, torch.Tensor]] = []
    for k, v in encoder_sd.items():
        if k.startswith("mlp.net.") and k.endswith(".weight"):
            # k format: mlp.net.{idx}.weight
            parts = k.split(".")
            try:
                idx = int(parts[2])
            except Exception:
                continue
            weights.append((idx, v))
    if not weights:
        raise KeyError("Cannot infer MLP structure: no `mlp.net.*.weight` found in encoder state_dict")
    weights.sort(key=lambda x: x[0])

    # Only Linear layer weights are used: shape [out, in]
    input_dim = int(weights[0][1].shape[1])
    layer_out_dims = [int(w.shape[0]) for _, w in weights]
    if len(layer_out_dims) < 1:
        raise ValueError("Failed to infer MLP structure: not enough Linear layers")
    hidden_dims = layer_out_dims[:-1]
    output_dim = layer_out_dims[-1]
    return input_dim, hidden_dims, output_dim


def _infer_vq_dims_from_state_dict(encoder_sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    w = encoder_sd.get("vq.embedding.weight", None)
    if w is None:
        raise KeyError("Cannot infer VQ structure: `vq.embedding.weight` not found in encoder state_dict")
    num_embeddings = int(w.shape[0])
    embedding_dim = int(w.shape[1])
    return num_embeddings, embedding_dim


def _load_motion_features(
    motion_path: str,
    motion_mode: str,
    *,
    npz_key: str | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    p = Path(motion_path)
    if not p.exists():
        raise FileNotFoundError(motion_path)

    if motion_mode == "npz_command":
        data = np.load(motion_path)
        if "joint_pos" not in data or "joint_vel" not in data:
            raise KeyError(f"npz_command requires joint_pos/joint_vel in npz. keys={list(data.keys())}")
        joint_pos = np.asarray(data["joint_pos"], dtype=dtype)
        joint_vel = np.asarray(data["joint_vel"], dtype=dtype)
        if joint_pos.ndim != 2 or joint_vel.ndim != 2:
            raise ValueError(f"joint_pos/joint_vel must be [T, D], got {joint_pos.shape} / {joint_vel.shape}")
        if joint_pos.shape[0] != joint_vel.shape[0]:
            raise ValueError(f"joint_pos/joint_vel have different T: {joint_pos.shape[0]} vs {joint_vel.shape[0]}")
        return np.concatenate([joint_pos, joint_vel], axis=-1)

    if motion_mode == "npz_command_anchor_identity":
        # In the tracking env, the encoder input often includes:
        #   command (= joint_pos/joint_vel) + motion_anchor_pos_b(3) + motion_anchor_ori_b(6).
        # Offline we don't have the robot state to compute "anchor relative to robot",
        # so we use a common approximation: robot_anchor == motion_anchor
        # => relative position is 0, relative orientation is Identity.
        # (Identity is represented by the first two columns of the rotation matrix flattened:
        #  [1,0,0, 0,1,0])
        data = np.load(motion_path)
        if "joint_pos" not in data or "joint_vel" not in data:
            raise KeyError(
                f"npz_command_anchor_identity requires joint_pos/joint_vel in npz. keys={list(data.keys())}"
            )
        joint_pos = np.asarray(data["joint_pos"], dtype=dtype)
        joint_vel = np.asarray(data["joint_vel"], dtype=dtype)
        if joint_pos.ndim != 2 or joint_vel.ndim != 2:
            raise ValueError(f"joint_pos/joint_vel must be [T, D], got {joint_pos.shape} / {joint_vel.shape}")
        if joint_pos.shape[0] != joint_vel.shape[0]:
            raise ValueError(f"joint_pos/joint_vel have different T: {joint_pos.shape[0]} vs {joint_vel.shape[0]}")
        T = joint_pos.shape[0]
        anchor_pos_b = np.zeros((T, 3), dtype=dtype)
        anchor_ori_b = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=dtype)[None, :], (T, 1))
        return np.concatenate([joint_pos, joint_vel, anchor_pos_b, anchor_ori_b], axis=-1)

    if motion_mode == "npz_key":
        if not npz_key:
            raise ValueError("When motion_mode=npz_key, you must provide --npz_key")
        data = np.load(motion_path)
        if npz_key not in data:
            raise KeyError(f"Key not found in npz: key={npz_key}. Available keys={list(data.keys())}")
        arr = np.asarray(data[npz_key], dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"npz_key={npz_key} must be [T, D], got {arr.shape}")
        return arr

    if motion_mode == "npy":
        arr = np.asarray(np.load(motion_path, allow_pickle=False), dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"npy must be [T, D], got {arr.shape}")
        return arr

    if motion_mode == "csv":
        arr = np.asarray(np.loadtxt(motion_path, delimiter=","), dtype=dtype)
        if arr.ndim != 2:
            raise ValueError(f"csv must be [T, D], got {arr.shape}")
        return arr

    raise ValueError(f"Unsupported motion_mode={motion_mode}")


def _stack_future_and_flatten(
    feats: np.ndarray,
    *,
    future_steps: int,
    future_stride: int,
    pad_mode: str,
) -> np.ndarray:
    """
    Expand input features into a "future K-step" window and flatten to 2D:
      - Input feats:
          - [T, D] or
          - [T, K, D] (will be directly flattened to [T, K*D], no re-stacking)
      - Output: [T, K*D]

    pad_mode:
      - repeat_last: out-of-range indices repeat the last frame
      - zero: out-of-range parts are padded with zeros
    """
    if feats.ndim == 3:
        # Already a future window: just flatten.
        T, K, D = feats.shape
        return feats.reshape(T, K * D)
    if feats.ndim != 2:
        raise ValueError(f"motion features must be [T,D] or [T,K,D], got {feats.shape}")

    if future_steps <= 1:
        return feats
    if future_steps < 1:
        raise ValueError(f"future_steps must be >= 1, got {future_steps}")
    if future_stride < 1:
        raise ValueError(f"future_stride must be >= 1, got {future_stride}")

    T, D = feats.shape
    # indices: [T, K]
    base = np.arange(T, dtype=np.int64)[:, None]
    offs = (np.arange(future_steps, dtype=np.int64) * future_stride)[None, :]
    idx = base + offs

    if pad_mode == "repeat_last":
        idx_clip = np.clip(idx, 0, T - 1)
        window = feats[idx_clip]  # [T, K, D]
        return window.reshape(T, future_steps * D)
    elif pad_mode == "zero":
        window = np.zeros((T, future_steps, D), dtype=feats.dtype)
        valid = idx < T
        # Fill element-wise (broadcast-friendly).
        for k in range(future_steps):
            valid_k = valid[:, k]
            if np.any(valid_k):
                window[valid_k, k, :] = feats[idx[valid_k, k]]
        return window.reshape(T, future_steps * D)
    else:
        raise ValueError(f"Unsupported pad_mode={pad_mode}. Options: repeat_last/zero")


def _compute_metrics(z_cont: torch.Tensor, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    z_cont/z_q: [T, D]
    Returns:
      - l2: [T]
      - cosine: [T]
      - abs_delta: [T, D]
    """
    if z_cont.shape != z_q.shape:
        raise ValueError(f"z_cont/z_q shape mismatch: {tuple(z_cont.shape)} vs {tuple(z_q.shape)}")
    delta = z_cont - z_q
    l2 = torch.linalg.norm(delta, dim=-1)
    # Cosine similarity
    zc = torch.nn.functional.normalize(z_cont, dim=-1, eps=1e-8)
    zq = torch.nn.functional.normalize(z_q, dim=-1, eps=1e-8)
    cosine = (zc * zq).sum(dim=-1)
    return {"l2": l2, "cosine": cosine, "abs_delta": delta.abs()}


def _plot_and_save(
    metrics: Dict[str, np.ndarray],
    out_dir: Path,
    *,
    fps: float | None = None,
    title: str = "VQ latent similarity",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    l2 = metrics["l2"]
    cosine = metrics["cosine"]
    abs_delta = metrics["abs_delta"]  # [T, D]
    t = np.arange(l2.shape[0], dtype=np.float32)
    x = t / fps if fps and fps > 0 else t
    x_label = "time (s)" if fps and fps > 0 else "frame"

    # 1) Curves: L2 + Cosine
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, l2, linewidth=1.2)
    ax1.set_title("L2 distance: ||z_cont - z_q||")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("L2")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, cosine, linewidth=1.2)
    ax2.set_title("Cosine similarity: cos(z_cont, z_q)")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("cosine")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "vq_similarity_curves.png", dpi=200)
    plt.close(fig)

    # 2) Heatmap: |delta|
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(abs_delta.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title("|z_cont - z_q| heatmap (dim x time)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("latent dim")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "vq_abs_delta_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_codebook_usage(
    encodings: np.ndarray,
    *,
    num_embeddings: int,
    out_dir: Path,
    title: str = "Codebook usage (counts per index)",
    max_bars: int | None = None,
) -> np.ndarray:
    """
    Codebook utilization:
      - x-axis: codebook index [0..num_embeddings-1]
      - y-axis: how many times this index is selected for this motion

    Returns counts: [num_embeddings]
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enc = np.asarray(encodings).reshape(-1)
    if enc.size == 0:
        raise ValueError("encodings is empty; cannot compute codebook utilization")
    if enc.dtype.kind not in ("i", "u"):
        # In some cases encodings might be float (shouldn't); force to int.
        enc = enc.astype(np.int64)
    if np.any(enc < 0) or np.any(enc >= num_embeddings):
        raise ValueError(
            f"encodings out of range: expected [0, {num_embeddings - 1}], "
            f"got min={enc.min()}, max={enc.max()}"
        )

    counts = np.bincount(enc, minlength=num_embeddings).astype(np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar plot (can be dense for large num_embeddings). Optionally restrict to top-N.
    if max_bars is not None and num_embeddings > max_bars:
        # Plot the top-N indices by count.
        top_idx = np.argsort(counts)[::-1][:max_bars]
        top_counts = counts[top_idx]
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(np.arange(top_idx.shape[0]), top_counts, width=0.9)
        ax.set_title(f"{title} (top {max_bars})")
        ax.set_xlabel("rank (sorted by count)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "codebook_usage_top.png", dpi=200)
        plt.close(fig)
    else:
        x = np.arange(num_embeddings, dtype=np.int64)
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x, counts, width=1.0)
        ax.set_title(title)
        ax.set_xlabel("codebook index")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "codebook_usage.png", dpi=200)
        plt.close(fig)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VQEncoder latent similarity/distance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pt checkpoint")
    parser.add_argument("--motion", type=str, required=True, help="Path to the motion file (npz/npy/csv)")
    parser.add_argument(
        "--motion_mode",
        type=str,
        required=True,
        choices=["npz_command", "npz_command_anchor_identity", "npz_key", "npy", "csv"],
        help="How to load motion features",
    )
    parser.add_argument("--npz_key", type=str, default=None, help="Used when motion_mode=npz_key")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (png + npz)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu/cuda:0, etc.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Forward in batches to avoid OOM")
    parser.add_argument("--fps", type=float, default=None, help="If provided, x-axis will be in seconds instead of frames")
    parser.add_argument("--no_plots", action="store_true", help="Only export npz, do not generate plots")
    parser.add_argument(
        "--future_steps",
        type=int,
        default=8,
        help="Expand features into a future K-step window (e.g., 8 means [t..t+7]) and flatten to [T, K*D]",
    )
    parser.add_argument("--future_stride", type=int, default=1, help="Stride within the future window (default: 1)")
    parser.add_argument(
        "--future_pad_mode",
        type=str,
        default="repeat_last",
        choices=["repeat_last", "zero"],
        help="Padding strategy when future indices go out of range",
    )
    parser.add_argument("--skip_dim_check", action="store_true", help="Skip feature dim vs encoder input_dim check")
    parser.add_argument("--skip_arch_check", action="store_true", help="Skip output_dim vs embedding_dim check (not recommended)")
    parser.add_argument(
        "--codebook_max_bars",
        type=int,
        default=None,
        help="If num_embeddings is large, plot only the top-N most frequent codes (writes codebook_usage_top.png)",
    )
    args = parser.parse_args()

    _add_repo_to_syspath()
    from rsl_rl.networks import VQEncoder  # noqa: E402

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # 1) Load checkpoint -> extract actor_encoder submodule weights
    model_sd = _load_checkpoint_state_dict(args.checkpoint, device)
    encoder_sd = _extract_sub_state_dict(model_sd, "actor_encoder.")

    # 2) Infer encoder architecture from weights and instantiate VQEncoder
    input_dim, hidden_dims, output_dim = _infer_mlp_dims_from_state_dict(encoder_sd)
    num_embeddings, embedding_dim = _infer_vq_dims_from_state_dict(encoder_sd)

    encoder = VQEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_weight=0.25,
        activation="elu",
    )
    # VectorQuantizer uses autocast(device_type="cuda") internally, so disable AMP on CPU.
    if hasattr(encoder, "vq") and hasattr(encoder.vq, "amp_enabled"):
        encoder.vq.amp_enabled = device.type == "cuda"

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        # strict=False is for robustness, but missing/unexpected keys likely indicate a mismatch.
        raise RuntimeError(f"Failed to load encoder weights: missing={missing}, unexpected={unexpected}")

    encoder.to(device).eval()

    # 3) Load motion features -> [T, D] (then optionally stack future window)
    feats_np = _load_motion_features(args.motion, args.motion_mode, npz_key=args.npz_key)
    feats_np = _stack_future_and_flatten(
        feats_np,
        future_steps=args.future_steps,
        future_stride=args.future_stride,
        pad_mode=args.future_pad_mode,
    )
    if (not args.skip_dim_check) and feats_np.shape[1] != input_dim:
        raise ValueError(
            f"Motion feature dim mismatch: current D={feats_np.shape[1]}, but encoder expects input_dim={input_dim}.\n"
            f"motion={args.motion} mode={args.motion_mode} future_steps={args.future_steps} stride={args.future_stride}\n"
            "Adjust --future_steps/--future_stride or motion_mode/npz_key, "
            "or preprocess motion into the same observation sub-vector used during training."
        )

    # 4) Forward pass to obtain z_cont (pre-VQ) and z_q (post-VQ)
    feats = torch.from_numpy(feats_np).to(device=device, dtype=torch.float32)
    zs_cont: List[torch.Tensor] = []
    zs_q: List[torch.Tensor] = []
    encodings_all: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, feats.shape[0], args.batch_size):
            x = feats[i : i + args.batch_size]
            z_q, _, _, encodings, z_cont = encoder(x)
            zs_cont.append(z_cont.detach().to("cpu"))
            zs_q.append(z_q.detach().to("cpu"))
            encodings_all.append(encodings.detach().to("cpu"))

    z_cont = torch.cat(zs_cont, dim=0)  # [T, D]
    z_q = torch.cat(zs_q, dim=0)  # [T, D]
    encodings = torch.cat(encodings_all, dim=0)  # [T]

    metrics_t = _compute_metrics(z_cont, z_q)
    counts = np.bincount(encodings.numpy().astype(np.int64), minlength=num_embeddings).astype(np.int64)
    used = int((counts > 0).sum())
    total = int(num_embeddings)
    coverage = used / float(total) if total > 0 else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Save raw outputs
    np.savez(
        out_dir / "vq_similarity_data.npz",
        z_cont=z_cont.numpy(),
        z_q=z_q.numpy(),
        encodings=encodings.numpy(),
        codebook_counts=counts,
        codebook_used=np.array([used], dtype=np.int64),
        codebook_total=np.array([total], dtype=np.int64),
        codebook_coverage=np.array([coverage], dtype=np.float32),
        l2=metrics_t["l2"].numpy(),
        cosine=metrics_t["cosine"].numpy(),
        abs_delta=metrics_t["abs_delta"].numpy(),
        fps=(np.array([args.fps], dtype=np.float32) if args.fps is not None else np.array([-1], dtype=np.float32)),
    )

    # 6) Plots
    if not args.no_plots:
        metrics_np: Dict[str, np.ndarray] = {k: v.detach().cpu().numpy() for k, v in metrics_t.items()}
        title = f"VQ similarity ({Path(args.motion).name})"
        _plot_and_save(metrics_np, out_dir, fps=args.fps, title=title)
        _plot_codebook_usage(
            encodings.numpy(),
            num_embeddings=num_embeddings,
            out_dir=out_dir,
            title=f"Codebook usage ({Path(args.motion).name})",
            max_bars=args.codebook_max_bars,
        )

    print(f"[OK] Outputs written to: {out_dir}")
    print(f" - data: {out_dir / 'vq_similarity_data.npz'}")
    print(f" - codebook coverage: {used}/{total} = {coverage:.4f}")
    if not args.no_plots:
        print(f" - plot: {out_dir / 'vq_similarity_curves.png'}")
        print(f" - plot: {out_dir / 'vq_abs_delta_heatmap.png'}")
        if args.codebook_max_bars is not None:
            print(f" - plot: {out_dir / 'codebook_usage_top.png'}")
        else:
            print(f" - plot: {out_dir / 'codebook_usage.png'}")


if __name__ == "__main__":
    main()


