# vehicle_8h_auto_model

Vehicle model auto annotation task in 8 hours.

Pipeline:

1. Split videos into frames at around 2 FPS
2. Run Moondrea VLM to detect cars and save in YOLO format
3. Merge labels with high IoU into 1
4. Convert dataset to COCO format
5. Train PP-YOLO model on it

Run split:

```bash
uv run split_video_to_frames.py --video "./dataset/train/videos" --frames-dir "./dataset/train/frames" --fps 2
```

Run auto annotation:

```bash
uv run auto_annotate.py --frames-dir "./dataset/train/frames" --labels-dir "./dataset/train/labels" --label "vehicle"
```

Merge labels with high IoU:

```bash
uv run merge_labels.py --labels-dir "./dataset/train/labels" --frames-dir "./dataset/train/frames" --iou-threshold 0.3
```

Convert to COCO:

```bash
uv run convert_to_coco.py --labels-dir "./dataset/train/labels" --images-dir "./dataset/train/frames" --destination-dir "./dataset/train/coco" --class-names "vehicle"
```

## Speedrun marks:

| Step                       | Time mark  |
| -------------------------- | ---------- |
| Split into frames complete | 20 minutes |
| Auto annotations complete  |            |

## Notes

### FOV assumptions

It's an HD video from drone, so it's an 16:9 camera, likely from Mavic. So diagonal FOV would be around 84 degree.
For a 16:9 video image, an 84° diagonal FOV corresponds approximately to: 76° horizontal × 48° vertical.

### Auto annotations

I wanted not to touch by hands much at all the training set, so there is functionality to merge labels with high IoU.

To review the dataset I use my own tool called Dataset GUI which I use to edit my datasets: ![Dataset GUI](./media/gui.png)

Sometimes the Moondream 2 VLM misses a little, like this: ![Miss](./media/miss.png), so I usually would clean it up by hand quickly in my editing tool: ![Editing](./media/editing.png)

But I want to try to train without cleanup for sake of the task. We could run some other model to do extra cleanup or rerun on each box as cut out, but it takes time and I expect training to take a while.

### Model

YOLO11 from Ultralytics has sh*t license, so using PP-YOLOE from paddle detection.

I picked S size, should be fast enough to train at given time. Input size was changed to 416x736px to move correspond to the videos aspect ratio. Train videos are horizontal, while Eval video is vertical so I rotated it into horizontal.

### GPU

I am bottlenecked by my GPU, but ideally you would run **Moondream 3** model instead of Moondrea 2 as it has better reasoning. Poor GPU: ![Poor GPU](./media/gpu.png)
