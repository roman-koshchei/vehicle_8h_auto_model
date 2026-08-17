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

| Step                                      | Time mark in 8 hours |
| ----------------------------------------- | -------------------- |
| Split into frames complete                | 20 minutes           |
| Auto annotations complete                 | 2 hours 20 minutes   |
| Functional cleanup, ready to run training | 3 hours 50 minutes   |
| First training run complete               | 5 hours 15 minutes   |

## Notes

### FOV assumptions

It's an HD video from drone, so it's an 16:9 camera, likely from Mavic. So diagonal FOV would be around 84 degree.
For a 16:9 video image, an 84° diagonal FOV corresponds approximately to: 76° horizontal × 48° vertical.

### Auto annotations

To review the dataset I use my own tool called Dataset GUI which I use to edit my datasets: ![Dataset GUI](./media/gui.png)

Sometimes the Moondream 2 VLM misses a little, like this: ![Miss](./media/miss.png), so I usually would clean it up by hand quickly in my editing tool: ![Editing](./media/editing.png)

Moondream 3 performs better but doesn't fit on my GPU.

I am running merge of labels with reasonable IoU.
And then I am doing light editing with my tool to remove big misses.
Like this:
![Big misses](./media/big_misses.png)

Unfortunately Moondream 2 didn't perform best on the high above video. Moondream 3 performs slightly better, but still does have some misses: ![moondream3 example](./media/moondream3_example.png.png)

One of the improvements was to take frame from far above, split it into 4 and run VLM on each and then combine boxes back into 1 big image space. That helped.

I also inflated small boxes by 10% to better contain small objects.

**NOTE**: after first training run I decided to try to do real clean up by hand to compare results, because mAP wasn't super great (but it's also because amount of data is small).

### Model

YOLO11 from Ultralytics has sh*t license, so using PP-YOLOE from paddle detection.

I picked S size, should be fast enough to train at given time. Input size was changed to 416x736px to move correspond to the videos aspect ratio. Train videos are horizontal, while Eval video is vertical so I rotated it into horizontal.

### GPU

I am bottlenecked by my GPU, but ideally you would run **Moondream 3** model instead of Moondrea 2 as it has better reasoning. Poor GPU: ![Poor GPU](./media/gpu.png)

## Paddle install

```bash
cd PaddleDetection
uv venv --python 3.9 .venv

# paddle 2.6.2 GPU wheel for CUDA 12.0 (cu126 index only ships paddle 3.x on Windows)
uv pip install --python ".venv\Scripts\python.exe" paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

# provides cudnn64_8.dll (v8.9) required by the cu120 wheel; add its bin dir to PATH
uv pip install --python ".venv\Scripts\python.exe" nvidia-cudnn-cu12==8.9.7.29
uv pip install --python ".venv\Scripts\python.exe" "setuptools<81"

# verify GPU: expect "PaddlePaddle works well on 1 GPU"
.venv\Scripts\python.exe -c "import paddle; paddle.utils.run_check()"

uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
.venv\Scripts\python.exe setup.py install
uv pip install --python ".venv\Scripts\python.exe" scikit-learn "numba==0.56.4"

# sanity test: expect 7/7 OK
.venv\Scripts\python.exe ppdet/modeling/tests/test_architectures.py
```
