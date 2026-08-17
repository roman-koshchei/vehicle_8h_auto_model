import argparse
import os

import cv2
from PIL import Image


class YoloLabel:
    def __init__(
        self,
        class_id: int,
        norm_x_center: float,
        norm_y_center: float,
        norm_width: float,
        norm_height: float,
    ):
        self.class_id = class_id
        self.norm_x_center = norm_x_center
        self.norm_y_center = norm_y_center
        self.norm_width = norm_width
        self.norm_height = norm_height


def write_file_labels(file_path: str, labels: list[YoloLabel]) -> None:
    with open(file_path, "w") as f:
        for label in labels:
            f.write(
                f"{label.class_id} {label.norm_x_center} {label.norm_y_center} {label.norm_width} {label.norm_height}\n"
            )


def detect(model, img, label: str):
    detected_objects = model.detect(img, label)["objects"]
    if not detected_objects:
        return []

    bounding_boxes = []
    for obj in detected_objects:
        x_min = obj["x_min"]
        y_min = obj["y_min"]
        x_max = obj["x_max"]
        y_max = obj["y_max"]

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        bounding_boxes.append((x_center, y_center, width, height))
    return bounding_boxes


def split_image_quadrants(pil_img):
    """
    Splits a PIL image into 4 quadrants. Returns a list of
    (quadrant_image, norm_x_offset, norm_y_offset) where the offsets are the
    top-left corner of each quadrant in normalized (0-1) coordinates.
    """
    width, height = pil_img.size
    mid_x = width // 2
    mid_y = height // 2

    return [
        (pil_img.crop((0, 0, mid_x, mid_y)), 0.0, 0.0),
        (pil_img.crop((mid_x, 0, width, mid_y)), 0.5, 0.0),
        (pil_img.crop((0, mid_y, mid_x, height)), 0.0, 0.5),
        (pil_img.crop((mid_x, mid_y, width, height)), 0.5, 0.5),
    ]


def remap_quadrant_label(
    label: YoloLabel, offset_x: float, offset_y: float
) -> YoloLabel:
    """
    Remaps a label detected in a quadrant (0.5-size image) back to full image
    normalized coordinates.
    """
    return YoloLabel(
        class_id=label.class_id,
        norm_x_center=offset_x + label.norm_x_center * 0.5,
        norm_y_center=offset_y + label.norm_y_center * 0.5,
        norm_width=label.norm_width * 0.5,
        norm_height=label.norm_height * 0.5,
    )


def detect_label_quadrants(model, pil_img, label: str) -> list[YoloLabel]:
    """
    Runs detection on each of the 4 quadrants of the image so small objects are
    relatively larger to the model, returning labels in full image normalized
    coordinates. Duplicates spanning quadrant boundaries are merged later.
    """
    all_detected = []
    for quadrant, offset_x, offset_y in split_image_quadrants(pil_img):
        for x, y, w, h in detect(model, quadrant, label):
            all_detected.append(
                remap_quadrant_label(
                    YoloLabel(
                        class_id=0, norm_x_center=x, norm_y_center=y, norm_width=w, norm_height=h
                    ),
                    offset_x,
                    offset_y,
                )
            )
    return all_detected


def auto_annotate(
    frames_directory_path: str,
    labels_directory_path: str,
    label: str,
    class_id: int = 0,
    split_quadrants: bool = False,
    name_prefix: str | None = None,
):
    if not os.path.isdir(frames_directory_path):
        raise ValueError(f"Not a valid directory: {frames_directory_path}")

    os.makedirs(labels_directory_path, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )
    model.compile()

    for frame_name in sorted(os.listdir(frames_directory_path)):
        if name_prefix is not None and not frame_name.startswith(name_prefix):
            continue

        frame_path = os.path.join(frames_directory_path, frame_name)
        if not os.path.isfile(frame_path):
            continue

        try:
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Error: Could not read frame {frame_path}")
                continue

            height, width = img.shape[:2]
            if width > 1920 or height > 1920:
                scale = min(1920 / width, 1920 / height)
                img = cv2.resize(
                    img,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            if split_quadrants:
                from merge_labels import filter_same_boxes_for_labels

                detected_labels = filter_same_boxes_for_labels(
                    img, detect_label_quadrants(model, pil_img, label), 0.5
                )
            else:
                detected_labels = [
                    YoloLabel(class_id=class_id, norm_x_center=x, norm_y_center=y, norm_width=w, norm_height=h)
                    for x, y, w, h in detect(model, pil_img, label)
                ]
            for detected_label in detected_labels:
                detected_label.class_id = class_id

            label_name = os.path.splitext(frame_name)[0] + ".txt"
            label_path = os.path.join(labels_directory_path, label_name)
            write_file_labels(label_path, detected_labels)
            print(f"Annotated: {frame_name} ({len(detected_labels)} objects)")

        except Exception as err:
            print(f"Error processing {frame_path}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate frames using the Moondream2 VLM"
    )
    parser.add_argument(
        "--frames-dir", required=True, help="Path to the directory containing frames"
    )
    parser.add_argument(
        "--labels-dir", required=True, help="Path to the directory for label files"
    )
    parser.add_argument(
        "--label", required=True, help="Object label to detect (e.g. 'car')"
    )
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id (default: 0)")
    parser.add_argument(
        "--split-quadrants",
        action="store_true",
        help="Split each frame into 4 quadrants before detection to improve small object recall",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Only process frames whose filename starts with this prefix",
    )
    args = parser.parse_args()

    auto_annotate(args.frames_dir, args.labels_dir, args.label, args.class_id, args.split_quadrants, args.name_prefix)


if __name__ == "__main__":
    main()