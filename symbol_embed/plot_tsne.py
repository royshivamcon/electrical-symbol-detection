"""Static + hover-thumbnail t-SNE for the three embedding arms."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from symbol_embed.config import Paths
from symbol_embed.embed import embed_crops


def _thumb_b64(path: str, size: int = 64) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _top_class_mask(labels: np.ndarray, k: int = 12) -> tuple[np.ndarray, list[int]]:
    counts = np.bincount(labels.astype(int))
    top = list(np.argsort(-counts)[:k])
    return np.isin(labels, top), top


def plot_static(
    xy: np.ndarray,
    labels: np.ndarray,
    label_to_name: dict[int, str],
    out_path: Path,
    *,
    title: str,
    top_k: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    keep, top = _top_class_mask(labels, top_k)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xy[~keep, 0], xy[~keep, 1], c="#bbbbbb", s=8, alpha=0.4, linewidths=0)
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(top):
        m = labels == lab
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=14,
            color=cmap(i % 20),
            label=label_to_name.get(int(lab), str(lab))[:40],
            linewidths=0,
            alpha=0.85,
        )
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hover_html(
    xy: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    label_to_name: dict[int, str],
    out_path: Path,
    *,
    title: str,
    max_points: int = 10000,
) -> None:
    """Write HTML with a side-panel thumbnail.

    Plotly tooltips strip ``<img>`` tags, so thumbs live in a JS array and a
    panel updates on ``plotly_hover`` instead.
    """
    import json

    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    n = len(xy)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        xy, labels = xy[idx], labels[idx]
        paths = [paths[i] for i in idx]

    thumbs = [_thumb_b64(p) for p in paths]
    names = [label_to_name.get(int(l), str(l)) for l in labels]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xy[:, 0].tolist(),
                y=xy[:, 1].tolist(),
                mode="markers",
                marker=dict(
                    size=7,
                    color=labels.astype(int).tolist(),
                    colorscale="Turbo",
                    showscale=False,
                ),
                customdata=np.arange(len(names)).tolist(),
                hovertemplate="%{text}<extra></extra>",
                text=names,
            )
        ]
    )
    fig.update_layout(
        title=title,
        width=780,
        height=700,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_json = fig.to_json()
    meta = [{"name": nm, "b64": t} for nm, t in zip(names, thumbs)]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>{title}</title>
<style>
  body {{ margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee; }}
  .wrap {{ display:flex; gap:12px; padding:12px; align-items:flex-start; }}
  #plot {{ flex:1; min-width:0; }}
  #panel {{
    width:180px; flex-shrink:0; background:#1c1c1c; border:1px solid #333;
    border-radius:8px; padding:12px; position:sticky; top:12px;
  }}
  #panel img {{ width:100%; height:auto; background:#fff; border-radius:4px; image-rendering:pixelated; }}
  #panel .name {{ font-size:12px; margin-top:8px; word-break:break-word; color:#ccc; }}
  #panel .hint {{ font-size:11px; color:#777; margin-bottom:8px; }}
</style>
</head><body>
<div class="wrap">
  <div id="plot"></div>
  <div id="panel">
    <div class="hint">Hover a point</div>
    <img id="thumb" alt="thumbnail" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"/>
    <div class="name" id="label">—</div>
  </div>
</div>
<script>{get_plotlyjs()}</script>
<script>
const META = {json.dumps(meta)};
const fig = {fig_json};
Plotly.newPlot('plot', fig.data, fig.layout, {{responsive:true, displayModeBar:true}});
const thumb = document.getElementById('thumb');
const label = document.getElementById('label');
document.getElementById('plot').on('plotly_hover', (ev) => {{
  const i = ev.points[0].customdata;
  const m = META[i];
  if (!m) return;
  thumb.src = 'data:image/png;base64,' + m.b64;
  label.textContent = m.name;
}});
</script>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)


def _find_best_ckpts(runs_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for arm in ("pretrained", "contrastive", "arcface"):
        arm_dir = runs_root / arm
        if not arm_dir.exists():
            continue
        cands = sorted(arm_dir.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        if cands:
            found[arm] = cands[-1]
    return found


def _fit_tsne(emb: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise SystemExit(
            "sklearn import failed (often numpy/scipy mismatch). "
            "Fix with: pip install 'numpy<2' 'scikit-learn' --force-reinstall\n"
            f"Original error: {exc}"
        ) from exc
    n = len(emb)
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(emb.astype(np.float32))


def run_tsne(
    rid: str,
    *,
    ckpt_map: dict[str, Path] | None = None,
    splitsplit: str = "val",
    perplexity: float = 30.0,
    seed: int = 0,
    paths: Paths | None = None,
) -> Path:
    paths = paths or Paths()
    crops_dir = paths.crops_dir(rid)
    if not crops_dir.exists():
        raise FileNotFoundError(f"no crops at {crops_dir} — export this rid first")
    ckpt_map = ckpt_map or _find_best_ckpts(paths.runs_dir())
    if not ckpt_map:
        raise FileNotFoundError(f"no checkpoints under {paths.runs_dir()}")

    out_dir = paths.viz_dir(rid)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for arm, ckpt in sorted(ckpt_map.items()):
        print(f"[tsne] {arm} ← {ckpt}")
        emb, lab, img_paths, meta = embed_crops(ckpt, crops_dir, split=splitsplit)
        label_to_name = {int(k): v for k, v in meta["label_to_name"].items()}
        if not label_to_name:
            label_to_name = {int(v): k for k, v in meta["name_to_label"].items()}

        xy = _fit_tsne(emb, perplexity=perplexity, seed=seed)

        np.savez(
            out_dir / f"{arm}_tsne.npz",
            xy=xy,
            labels=lab,
            paths=np.array(img_paths),
        )
        plot_static(
            xy,
            lab,
            label_to_name,
            out_dir / f"{arm}_tsne.png",
            title=f"{arm} t-SNE (rid={rid[:8]})",
        )
        plot_hover_html(
            xy,
            lab,
            img_paths,
            label_to_name,
            out_dir / f"{arm}_tsne_hover.html",
            title=f"{arm} t-SNE (hover thumbnails)",
        )
        panels.append((arm, xy, lab, img_paths, label_to_name))

    if len(panels) >= 2:
        _write_compare_hover(panels, out_dir / "compare_tsne_hover.html", rid=rid)

    print(f"[tsne] wrote → {out_dir}")
    return out_dir


def _write_compare_hover(
    panels: list[tuple],
    out_path: Path,
    *,
    rid: str,
    max_points: int = 10000,
) -> None:
    """Multi-panel compare HTML with shared side thumbnail panel."""
    import json

    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels]
    )
    all_meta: list[dict] = []
    for col, (arm, xy, lab, img_paths, label_to_name) in enumerate(panels, start=1):
        n = min(len(xy), max_points)
        idx = (
            np.linspace(0, len(xy) - 1, n, dtype=int)
            if len(xy) > n
            else np.arange(len(xy))
        )
        base = len(all_meta)
        custom = []
        for j, i in enumerate(idx):
            nm = label_to_name.get(int(lab[i]), str(lab[i]))
            all_meta.append({"name": f"[{arm}] {nm}", "b64": _thumb_b64(img_paths[i])})
            custom.append(base + j)
        fig.add_trace(
            go.Scatter(
                x=xy[idx, 0].tolist(),
                y=xy[idx, 1].tolist(),
                mode="markers",
                marker=dict(
                    size=6,
                    color=lab[idx].astype(int).tolist(),
                    colorscale="Turbo",
                    showscale=False,
                ),
                customdata=custom,
                text=[label_to_name.get(int(lab[i]), str(lab[i])) for i in idx],
                hovertemplate="%{text}<extra></extra>",
                name=arm,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(visible=False, row=1, col=col)
        fig.update_yaxes(visible=False, row=1, col=col)
    fig.update_layout(
        title=f"t-SNE compare rid={rid[:8]}",
        height=650,
        width=360 * len(panels) + 200,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>t-SNE compare</title>
<style>
  body {{ margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee; }}
  .wrap {{ display:flex; gap:12px; padding:12px; align-items:flex-start; }}
  #plot {{ flex:1; min-width:0; }}
  #panel {{
    width:180px; flex-shrink:0; background:#1c1c1c; border:1px solid #333;
    border-radius:8px; padding:12px; position:sticky; top:12px;
  }}
  #panel img {{ width:100%; height:auto; background:#fff; border-radius:4px; image-rendering:pixelated; }}
  #panel .name {{ font-size:12px; margin-top:8px; word-break:break-word; color:#ccc; }}
  #panel .hint {{ font-size:11px; color:#777; margin-bottom:8px; }}
</style>
</head><body>
<div class="wrap">
  <div id="plot"></div>
  <div id="panel">
    <div class="hint">Hover a point</div>
    <img id="thumb" alt="thumbnail" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'/%3E"/>
    <div class="name" id="label">—</div>
  </div>
</div>
<script>{get_plotlyjs()}</script>
<script>
const META = {json.dumps(all_meta)};
const fig = {fig.to_json()};
Plotly.newPlot('plot', fig.data, fig.layout, {{responsive:true}});
const thumb = document.getElementById('thumb');
const label = document.getElementById('label');
document.getElementById('plot').on('plotly_hover', (ev) => {{
  const i = ev.points[0].customdata;
  const m = META[i];
  if (!m) return;
  thumb.src = 'data:image/png;base64,' + m.b64;
  label.textContent = m.name;
}});
</script>
</body></html>
"""
    out_path.write_text(html)


def rebuild_hover_from_npz(viz_dir: Path, label_to_name: dict[int, str] | None = None) -> None:
    """Regenerate hover HTML from existing ``*_tsne.npz`` (no re-embed)."""
    viz_dir = Path(viz_dir)
    panels = []
    for npz_path in sorted(viz_dir.glob("*_tsne.npz")):
        arm = npz_path.name.replace("_tsne.npz", "")
        data = np.load(npz_path, allow_pickle=True)
        xy = data["xy"]
        lab = data["labels"]
        paths = [str(p) for p in data["paths"].tolist()]
        if label_to_name is None:
            # recover names from path parent folder slug only as fallback
            l2n = {int(l): str(l) for l in set(lab.tolist())}
        else:
            l2n = label_to_name
        plot_hover_html(
            xy,
            lab,
            paths,
            l2n,
            viz_dir / f"{arm}_tsne_hover.html",
            title=f"{arm} t-SNE (hover thumbnails)",
        )
        panels.append((arm, xy, lab, paths, l2n))
        print(f"[rebuild] {arm}_tsne_hover.html")
    if len(panels) >= 2:
        rid = viz_dir.name
        _write_compare_hover(panels, viz_dir / "compare_tsne_hover.html", rid=rid)
        print("[rebuild] compare_tsne_hover.html")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rid", required=True, help="Single request id to visualise")
    ap.add_argument("--split", default="val", choices=("train", "val", "all"))
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rebuild-hover",
        action="store_true",
        help="Only regenerate hover HTML from existing *_tsne.npz (fast)",
    )
    ap.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="arm=/path/to/best.pt (repeatable). Default: latest under runs/",
    )
    args = ap.parse_args(argv)

    if args.rebuild_hover:
        paths = Paths()
        viz_dir = paths.viz_dir(args.rid)
        crops = paths.crops_dir(args.rid)
        l2n: dict[int, str] = {}
        splits_path = crops / "splits.json"
        if splits_path.exists():
            import json

            splits = json.loads(splits_path.read_text())
            l2n = {int(v): k for k, v in splits["name_to_label"].items()}
        rebuild_hover_from_npz(viz_dir, label_to_name=l2n or None)
        return

    ckpt_map = None
    if args.ckpt:
        ckpt_map = {}
        for item in args.ckpt:
            if "=" not in item:
                raise SystemExit(f"--ckpt needs arm=path, got {item!r}")
            arm, path = item.split("=", 1)
            ckpt_map[arm] = Path(path)
    run_tsne(
        args.rid,
        ckpt_map=ckpt_map,
        splitsplit=args.split,
        perplexity=args.perplexity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
