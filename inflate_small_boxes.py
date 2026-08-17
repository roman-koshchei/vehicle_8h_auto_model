import argparse
import os

import cv2

from auto_annotate import YoloLabel, write_file_labels
from merge_labels import read_file_labels

image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def box_pixel_dims(label, image) -> tuple[float, float]:
    img_height, img_width = image.shape[:2]
    return label.norm_width * img_width, label.norm_height * img_height


def is_small(label, image, min_dim: float) -> bool:
    box_width, box_height = box_pixel_dims(label, image)
    return min(box_width, box_height) < min_dim


def inflate_label(label: YoloLabel, scale: float) -> YoloLabel:
    return YoloLabel(
        class_id=label.class_id,
        norm_x_center=label.norm_x_center,
        norm_y_center=label.norm_y_center,
        norm_width=label.norm_width * scale,
        norm_height=label.norm_height * scale,
    )


def clamp_label_to_image(label: YoloLabel, image) -> YoloLabel:
    """
    Clamps normalized box dimensions so the box stays fully inside the image,
    keeping the center fixed.
    """
    max_width = 2 * min(label.norm_x_center, 1 - label.norm_x_center)
    max_height = 2 * min(label.norm_y_center, 1 - label.norm_y_center)

    return YoloLabel(
        class_id=label.class_id,
        norm_x_center=label.norm_x_center,
        norm_y_center=label.norm_y_center,
        norm_width=min(label.norm_width, max_width),
        norm_height=min(label.norm_height, max_height),
    )


def inflate_small_boxes(
    labels_directory_path: str,
    frames_directory_path: str,
    min_dim: float = 64,
    scale: float = 1.1,
    out_directory_path: str | None = None,
):
    if not os.path.isdir(labels_directory_path):
        raise ValueError(f"Not a valid directory: {labels_directory_path}")
    if not os.path.isdir(frames_directory_path):
        raise ValueError(f"Not a valid directory: {frames_directory_path}")

    if out_directory_path is not None:
        os.makedirs(out_directory_path, exist_ok=True)

    label_files = sorted(f for f in os.listdir(labels_directory_path) if f.endswith(".txt"))

    total_inflated = 0
    boxes_changed = 0

    for label_name in label_files:
        label_path = os.path.join(labels_directory_path, label_name)
        labels = read_file_labels(label_path)

        frame_name = os.path.splitext(label_name)[0]
        frame_path = None
        for ext in image_exts:
            candidate = os.path.join(frames_directory_path, frame_name + ext)
            if os.path.exists(candidate):
                frame_path = candidate
                break

        changed = False
        if frame_path is not None:
            image = cv2.imread(frame_path)
            if image is not None and labels:
                for i, label in enumerate(labels):
                    if not is_small(label, image, min_dim):
                        continue
                    inflated = clamp_label_to_image(inflate_label(label, scale), image)
                    if (inflated.norm_width, inflated.norm_height) != (label.norm_width, label.norm_height):
                        labels[i] = inflated
                        changed = True
                        total_inflated += 1

        if changed:
            boxes_changed += 1

        if out_directory_path is not None:
            write_file_labels(os.path.join(out_directory_path, label_name), labels)
        elif changed:
            write_file_labels(label_path, labels)

    print(f"Processed {len(label_files)} label files")
    if label_files:
        print(
            f"Boxes inflated (min side < {min_dim}px) by {scale}x: {total_inflated}, "
            f"files changed: {boxes_changed}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Increase the size of small boxes in a YOLO dataset by a given scale"
    )
    parser.add_argument(
        "--labels-dir", required=True, help="Path to the directory containing YOLO label files"
    )
    parser.add_argument(
        "--frames-dir", required=True, help="Path to the directory containing image frames"
    )
    parser.add_argument(
        "--min-dim",
        type=float,
        default=64,
        help="Boxes with a side shorter than this (in pixels) are considered small (default: 64)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="Factor to increase small box dimensions by (default: 1.1)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for the new label files (default: overwrite in place)",
    )
    args = parser.parse_args()

    inflate_small_boxes(
        args.labels_dir,
        args.frames_dir,
        args.min_dim,
        args.scale,
        args.out_dir,
    )


if __name__ == "__main__":
    main()