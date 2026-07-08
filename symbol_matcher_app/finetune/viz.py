"""Draw FastSAM candidate masks on a point's crop, colored by label + head score.

Uses the cached ``dataset/crops/*.npz`` (grayscale crop + packed masks + prompt point)
so it never re-renders or re-runs FastSAM. Each candidate mask is blended onto the crop
(green = positive by the rule, red = negative), outlined, and annotated with the learned
head score ``h`` and the baseline FastSAM objectness ``c``. The prompt point is marked.

    ../.envs/vsam/bin/python finetune/viz.py --wid 78302e4e            # first point of a sheet
    ../.envs/vsam/bin/python finetune/viz.py --wid 78302e4e --pt 3     # a specific point
    ../.envs/vsam/bin/python finetune/viz.py --wid 78302e4e --pos-only # only the positive mask(s)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent
for _p in (str(APP_DIR), str(FT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import CKPT_PATH, MANIFEST  # noqa: E402
from dataset import _load_npz  # noqa: E402

POS_BGR = (80, 200, 80)    # green
NEG_BGR = (60, 60, 220)    # red
PT_BGR = (255, 255, 0)     # cyan


def _pick_point(rows: list[dict], rid: str | None, wid: str | None, pt: int | None) -> str:
    """Return the npz filename for the chosen point (first matching, preferring one
    that has a positive candidate) or raise if nothing matches."""
    cand = [r for r in rows
            if (not rid or r["rid"].startswith(rid)) and (not wid or r["wid"].startswith(wid))]
    if not cand:
        raise SystemExit(f"[viz] no rows for rid={rid} wid={wid}")
    if pt is not None:
        cand = [r for r in cand if r["pt_idx"] == pt]
        if not cand:
            raise SystemExit(f"[viz] no rows for pt_idx={pt}")
        return cand[0]["npz"]
    with_pos = {r["npz"] for r in cand if r["label"] == 1}
    for r in cand:
        if r["npz"] in with_pos:
            return r["npz"]
    return cand[0]["npz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--ckpt", type=str, default=str(CKPT_PATH))
    ap.add_argument("--rid", type=str, default=None, help="request id (prefix ok)")
    ap.add_argument("--wid", type=str, default=None, help="sheet id (prefix ok)")
    ap.add_argument("--pt", type=int, default=None, help="pt_idx (default: first w/ a positive)")
    ap.add_argument("--pos-only", action="store_true", help="draw only positive-labeled masks")
    ap.add_argument("--out", type=str, default=None, help="output PNG path")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows = [json.loads(l) for l in manifest.open()]
    npz = _pick_point(rows, args.rid, args.wid, args.pt)
    prows = sorted((r for r in rows if r["npz"] == npz), key=lambda r: r["k"])
    gray, masks, cx, cy = _load_npz(str(manifest.parent / "crops" / npz))

    scorer = None
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        from infer import MaskConfidenceScorer
        scorer = MaskConfidenceScorer.load(ckpt)
        head = scorer.score_many(gray, [masks[r["k"]] for r in prows], cx, cy,
                                  confs=[r["conf"] for r in prows])
    else:
        head = [None] * len(prows)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r, hs in zip(prows, head):
        if args.pos_only and r["label"] != 1:
            continue
        m = masks[r["k"]]
        color = POS_BGR if r["label"] == 1 else NEG_BGR
        overlay = img.copy()
        overlay[m] = color
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 1)
        ys, xs = np.where(m)
        tag = f"c={r['conf']:.2f}" + (f" h={hs:.2f}" if hs is not None else "")
        cv2.putText(img, tag, (int(xs.min()), max(10, int(ys.min()) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.drawMarker(img, (cx, cy), PT_BGR, cv2.MARKER_CROSS, 18, 2)
    cv2.circle(img, (cx, cy), 4, PT_BGR, 1, cv2.LINE_AA)

    out = Path(args.out) if args.out else (manifest.parent / "viz" / (Path(npz).stem + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    r0 = prows[0]
    n_pos = sum(r["label"] for r in prows)
    print(f"[viz] {r0['rid'][:8]}/{r0['wid'][:8]} pt={r0['pt_idx']} '{r0['name']}' "
          f"masks={len(prows)} (pos={n_pos}) crop={img.shape[1]}x{img.shape[0]}")
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
