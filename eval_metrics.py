import argparse
import glob
import json
import math
import os

IMAGE_W, IMAGE_H = 1280.0, 720.0
FOV_DIAG_DEG = 84.0
FOV_H_DEG = 76.0
CAR_LENGTH_M = 4.5
FOCAL_PX = (IMAGE_W / 2.0) / math.tan(math.radians(FOV_H_DEG / 2.0))
IOU_THRESHOLD = 0.5
N_FRAMES = 61
FPS_SAMPLE = 2.0
BANDS = [(0.0, 200.0, "0-200 m"), (200.0, 400.0, "200-400 m")]


def estimate_distance(px_size):
    if px_size <= 0:
        return float("inf")
    return CAR_LENGTH_M * FOCAL_PX / px_size


def load_ground_truth(labels_dir):
    gt = []
    for path in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
        frame = int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0])
        for line in open(path):
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(x) for x in parts)
            x = (cx - w / 2.0) * IMAGE_W
            y = (cy - h / 2.0) * IMAGE_H
            bw = w * IMAGE_W
            bh = h * IMAGE_H
            gt.append({"frame": frame, "box": [x, y, bw, bh]})
    return gt


def load_detections(bbox_json_path):
    dets = json.load(open(bbox_json_path))
    out = []
    for d in dets:
        x, y, w, h = d["bbox"]
        out.append({"frame": d["image_id"], "score": d["score"], "box": [x, y, w, h]})
    return out


def clip_box(box):
    x, y, w, h = box
    x2, y2 = min(x + w, IMAGE_W), min(y + h, IMAGE_H)
    x, y = max(x, 0.0), max(y, 0.0)
    return [x, y, max(x2 - x, 0.0), max(y2 - y, 0.0)]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match(dets, gt, threshold):
    dets = [d for d in dets if d["score"] >= threshold]
    dets = sorted(dets, key=lambda d: -d["score"])
    matched_gt = set()
    tp_dets = []
    fp_dets = []
    for d in dets:
        d["box"] = clip_box(d["box"])
        best_iou = 0.0
        best_gt = None
        for gi, g in enumerate(gt):
            if g["frame"] != d["frame"] or gi in matched_gt:
                continue
            o = iou(d["box"], g["box"])
            if o > best_iou:
                best_iou = o
                best_gt = gi
        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt)
            tp_dets.append((d, best_gt))
        else:
            fp_dets.append(d)
    return tp_dets, fp_dets, matched_gt


def det_dist(d):
    return estimate_distance(max(d["box"][2], d["box"][3]))


def band_of(dist):
    for lo, hi, name in BANDS:
        if lo < dist <= hi:
            return name
    return "outside"


def main():
    parser = argparse.ArgumentParser(description="Distance-band metrics for the v1 model")
    parser.add_argument("--labels-dir", default="dataset/eval/labels")
    parser.add_argument("--bbox-json", default="eval_bbox/bbox.json")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()

    gt = load_ground_truth(args.labels_dir)
    dets = load_detections(args.bbox_json)

    print(f"Assumptions: car length={CAR_LENGTH_M}m, FOV={FOV_DIAG_DEG}deg diag "
          f"({FOV_H_DEG}deg H), focal={FOCAL_PX:.1f}px, IoU>={IOU_THRESHOLD}, "
          f"{len(args.thresholds)} score thresholds, N_frames={N_FRAMES} @{FPS_SAMPLE:g}fps")
    print(f"GT boxes: {len(gt)}  Detections: {len(dets)}")
    print()

    dists = [estimate_distance(max(g["box"][2], g["box"][3])) for g in gt]
    finite_dists = sorted(d for d in dists if math.isfinite(d))
    if finite_dists:
        print(f"GT distance (m): min={finite_dists[0]:.0f} "
              f"median={finite_dists[len(finite_dists)//2]:.0f} max={finite_dists[-1]:.0f}")

    for band_lo, band_hi, band_name in BANDS:
        band_gt = [g for g, d in zip(gt, dists) if band_lo < d <= band_hi]
        if not band_gt:
            print()
            print(f"== {band_name}: NO GROUND TRUTH, metrics = N/A ==")
            continue
        gt_ids = set(id(g) for g in band_gt)
        print()
        print(f"== {band_name}: {len(band_gt)} GT boxes ==")
        header = f"{'Thr':>4} {'DetRate':>9} {'Precision':>9} {'FA/min':>9} {'T2F(s)':>8}"
        print(header)
        for thr in args.thresholds:
            tp_dets, fp_dets, matched = match(dets, gt, thr)
            band_tp = sum(1 for gi in matched if id(gt[gi]) in gt_ids)
            band_fp = sum(1 for d in fp_dets if band_lo < det_dist(d) <= band_hi)
            band_fn = len(band_gt) - band_tp
            det_rate = band_tp / (band_tp + band_fn) if (band_tp + band_fn) else float("nan")
            precision = band_tp / (band_tp + band_fp) if (band_tp + band_fp) else float("nan")
            fa_min = band_fp * 60.0 / N_FRAMES
            tp_frames = sorted({d["frame"] for d, gi in tp_dets if id(gt[gi]) in gt_ids})
            t2f = (tp_frames[0] - 1) / FPS_SAMPLE if tp_frames else float("inf")
            t2f_s = f"{t2f:.1f}" if math.isfinite(t2f) else "inf"
            print(f"{thr:>4.1f} {det_rate:>9.3f} {precision:>9.3f} {fa_min:>9.2f} {t2f_s:>8}")


if __name__ == "__main__":
    main()