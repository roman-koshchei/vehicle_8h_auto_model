import argparse
import os

import cv2

from auto_annotate import YoloLabel, write_file_labels

image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def read_file_labels(file_path: str) -> list[YoloLabel]:
    labels: list[YoloLabel] = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                label = YoloLabel(
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                labels.append(label)
    return labels


def iou(image, label_1: YoloLabel, label_2: YoloLabel) -> float:
    """
    Calculates Intersection over Union for 2 YOLO labels with normalized coordinates.
    Requires image to convert normalized (0-1) values to pixel ones.
    """
    img_height, img_width = image.shape[:2]

    x1_center = label_1.norm_x_center * img_width
    y1_center = label_1.norm_y_center * img_height
    w1 = label_1.norm_width * img_width
    h1 = label_1.norm_height * img_height

    x1_min = x1_center - w1 / 2
    y1_min = y1_center - h1 / 2
    x1_max = x1_center + w1 / 2
    y1_max = y1_center + h1 / 2

    x2_center = label_2.norm_x_center * img_width
    y2_center = label_2.norm_y_center * img_height
    w2 = label_2.norm_width * img_width
    h2 = label_2.norm_height * img_height

    x2_min = x2_center - w2 / 2
    y2_min = y2_center - h2 / 2
    x2_max = x2_center + w2 / 2
    y2_max = y2_center + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    area_1 = w1 * h1
    area_2 = w2 * h2
    union_area = area_1 + area_2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def merge_labels_into_single(
    label_1: YoloLabel, label_2: YoloLabel
) -> YoloLabel:
    """
    Merges two YOLO labels by averaging their coordinates and taking the larger box dimensions.
    Uses the class_id from the first label.
    """
    merged_x_center = (label_1.norm_x_center + label_2.norm_x_center) / 2
    merged_y_center = (label_1.norm_y_center + label_2.norm_y_center) / 2

    merged_width = max(label_1.norm_width, label_2.norm_width)
    merged_height = max(label_1.norm_height, label_2.norm_height)

    return YoloLabel(
        class_id=label_1.class_id,
        norm_x_center=merged_x_center,
        norm_y_center=merged_y_center,
        norm_width=merged_width,
        norm_height=merged_height,
    )


def filter_same_boxes_for_labels(
    image, labels: list[YoloLabel], iou_threshold: float = 0.5
) -> list[YoloLabel]:
    """
    Filters out overlapping boxes with same class using Non-Max Suppression approach.
    Boxes with IoU > iou_threshold and same class_id are merged into a single box.
    """
    if not labels:
        return []

    class_groups = {}
    for label in labels:
        if label.class_id not in class_groups:
            class_groups[label.class_id] = []
        class_groups[label.class_id].append(label)

    filtered_labels = []

    for class_id, class_labels in class_groups.items():
        if len(class_labels) == 1:
            filtered_labels.extend(class_labels)
            continue

        class_labels.sort(
            key=lambda x: x.norm_width * x.norm_height, reverse=True
        )

        processed = [False] * len(class_labels)

        for i in range(len(class_labels)):
            if processed[i]:
                continue

            current_label = class_labels[i]
            labels_to_merge = [current_label]
            processed[i] = True

            for j in range(i + 1, len(class_labels)):
                if processed[j]:
                    continue

                if iou(image, current_label, class_labels[j]) > iou_threshold:
                    labels_to_merge.append(class_labels[j])
                    processed[j] = True

            merged_label = labels_to_merge[0]
            for label_to_merge in labels_to_merge[1:]:
                merged_label = merge_labels_into_single(merged_label, label_to_merge)

            filtered_labels.append(merged_label)

    return filtered_labels


def merge_labels(
    labels_directory_path: str,
    frames_directory_path: str,
    iou_threshold: float = 0.5,
):
    if not os.path.isdir(labels_directory_path):
        raise ValueError(f"Not a valid directory: {labels_directory_path}")
    if not os.path.isdir(frames_directory_path):
        raise ValueError(f"Not a valid directory: {frames_directory_path}")

    label_files = sorted(f for f in os.listdir(labels_directory_path) if f.endswith(".txt"))

    total_original = 0
    total_merged = 0
    files_changed = 0

    for label_name in label_files:
        label_path = os.path.join(labels_directory_path, label_name)
        labels = read_file_labels(label_path)

        if len(labels) <= 1:
            continue

        frame_name = os.path.splitext(label_name)[0]
        frame_path = None
        for ext in image_exts:
            candidate = os.path.join(frames_directory_path, frame_name + ext)
            if os.path.exists(candidate):
                frame_path = candidate
                break

        if not frame_path:
            continue

        image = cv2.imread(frame_path)
        if image is None:
            continue

        total_original += len(labels)
        merged = filter_same_boxes_for_labels(image, labels, iou_threshold)
        total_merged += len(merged)

        if len(merged) != len(labels):
            write_file_labels(label_path, merged)
            files_changed += 1
            print(
                f"Merged {label_name}: {len(labels)} -> {len(merged)} boxes"
            )

    print(f"Processed {len(label_files)} label files")
    if label_files:
        print(
            f"Total boxes: {total_original} -> {total_merged}, "
            f"files changed: {files_changed}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Merge overlapping same-class labels with high IoU in a YOLO dataset"
    )
    parser.add_argument(
        "--labels-dir", required=True, help="Path to the directory containing YOLO label files"
    )
    parser.add_argument(
        "--frames-dir", required=True, help="Path to the directory containing image frames"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for merging same-class boxes (default: 0.5)",
    )
    args = parser.parse_args()

    merge_labels(args.labels_dir, args.frames_dir, args.iou_threshold)


if __name__ == "__main__":
    main()
