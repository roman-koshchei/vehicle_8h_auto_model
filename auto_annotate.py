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


def auto_annotate(
    frames_directory_path: str,
    labels_directory_path: str,
    label: str,
    class_id: int = 0,
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

            detected_labels = [
                YoloLabel(class_id=class_id, norm_x_center=x, norm_y_center=y, norm_width=w, norm_height=h)
                for x, y, w, h in detect(model, pil_img, label)
            ]

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
    args = parser.parse_args()

    auto_annotate(args.frames_dir, args.labels_dir, args.label, args.class_id)


if __name__ == "__main__":
    main()