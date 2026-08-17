import argparse
import json
import os
import shutil

import cv2

from auto_annotate import YoloLabel


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


def read_file_labels_with_fallback(file_path: str) -> list[YoloLabel]:
    if os.path.exists(file_path):
        return read_file_labels(file_path)
    else:
        return []


def label_name_from_image_name(image_name: str):
    name = os.path.splitext(image_name)[0]
    return f"{name}.txt"


def files_in_dir(dir: str) -> list[tuple[str, str]]:
    """Lists file name and file path in directory sorted by name in ascending order"""
    filenames = sorted(os.listdir(dir))
    return [(filename, os.path.join(dir, filename)) for filename in filenames]


def convert_yolo_to_coco_dataset(
    labels_dir: str,
    images_dir: str,
    destination_dir: str,
    class_names: list[str] | None = None,
):
    destination_images_dir = os.path.join(destination_dir, "images")
    destination_annotations_dir = os.path.join(destination_dir, "annotations")
    destination_annotations_path = os.path.join(
        destination_annotations_dir, "instances.json"
    )

    os.makedirs(destination_images_dir, exist_ok=True)
    os.makedirs(destination_annotations_dir, exist_ok=True)

    images = []
    annotations = []

    if class_names is not None:
        class_id_to_category_id = {idx: idx + 1 for idx in range(len(class_names))}
    else:
        class_id_to_category_id: dict[int, int] = {}

    annotation_id = 1
    image_id = 1

    image_files = files_in_dir(images_dir)
    total_images = len(image_files)

    for idx, (image_name, image_path) in enumerate(image_files, 1):
        if idx % 100 == 0:
            print(f"Processing [{idx}/{total_images}]: {image_name}")

        if not os.path.isfile(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        label_path = os.path.join(labels_dir, label_name_from_image_name(image_name))
        labels = read_file_labels_with_fallback(label_path)

        shutil.copy2(image_path, os.path.join(destination_images_dir, image_name))

        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
            }
        )

        for label in labels:
            if class_names is not None:
                if label.class_id not in class_id_to_category_id:
                    continue
            else:
                if label.class_id not in class_id_to_category_id:
                    class_id_to_category_id[label.class_id] = (
                        len(class_id_to_category_id) + 1
                    )

            x_center = label.norm_x_center * width
            y_center = label.norm_y_center * height
            bbox_width = label.norm_width * width
            bbox_height = label.norm_height * height

            x_min = max(0.0, x_center - bbox_width / 2)
            y_min = max(0.0, y_center - bbox_height / 2)
            x_max = min(float(width), x_center + bbox_width / 2)
            y_max = min(float(height), y_center + bbox_height / 2)

            bbox_width = max(0.0, x_max - x_min)
            bbox_height = max(0.0, y_max - y_min)

            if bbox_width == 0 or bbox_height == 0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id_to_category_id[label.class_id],
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "segmentation": [
                        [
                            x_min,
                            y_min,
                            x_min + bbox_width,
                            y_min,
                            x_min + bbox_width,
                            y_min + bbox_height,
                            x_min,
                            y_min + bbox_height,
                        ]
                    ],
                }
            )
            annotation_id += 1

        image_id += 1

    categories = [
        {
            "id": category_id,
            "name": f"class_{yolo_class_id}",
            "supercategory": "object",
            "yolo_class_id": yolo_class_id,
        }
        for yolo_class_id, category_id in sorted(
            class_id_to_category_id.items(), key=lambda item: item[1]
        )
    ]

    if class_names is not None:
        for category in categories:
            yolo_class_id = category["yolo_class_id"]
            category["name"] = class_names[yolo_class_id]

    coco_dataset = {
        "info": {
            "description": "Converted from YOLO dataset",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(destination_annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, indent=2)

    print(f"COCO dataset created in: {destination_dir}")
    print(f"Images copied: {len(images)}")
    print(f"Annotations written: {len(annotations)}")
    print(f"Categories written: {len(categories)}")
    print(f"Annotations file: {destination_annotations_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-format dataset to COCO format"
    )
    parser.add_argument("--labels-dir", required=True, help="Path to the labels directory")
    parser.add_argument("--images-dir", required=True, help="Path to the images directory")
    parser.add_argument(
        "--destination-dir", required=True, help="Path to the output COCO dataset directory"
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Class names ordered by YOLO class id (optional)",
    )
    args = parser.parse_args()

    convert_yolo_to_coco_dataset(
        args.labels_dir,
        args.images_dir,
        args.destination_dir,
        args.class_names,
    )


if __name__ == "__main__":
    main()