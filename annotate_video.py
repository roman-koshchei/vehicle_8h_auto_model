import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional

import cv2

PADDLEDETECTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PaddleDetection")


def split_video_to_frames(video_path: str, frames_dir: str, fps: Optional[float] = None) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    filter_args = ["-vf", f"fps={fps}"] if fps is not None else []
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        *filter_args,
        "-y",
        os.path.join(frames_dir, "frame_%05d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract frames: {result.stderr}")


def run_inference(config: str, weights: str, frames_dir: str, output_dir: str, draw_threshold: float) -> None:
    infer_py = os.path.join(PADDLEDETECTION_DIR, "tools", "infer.py")
    cmd = [
        sys.executable,
        infer_py,
        "-c",
        config,
        "-o",
        f"weights={weights}",
        "--infer_dir",
        frames_dir,
        "--output_dir",
        output_dir,
        "--draw_threshold",
        str(draw_threshold),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Inference failed")


def annotated_frames_to_video(annotated_dir: str, out_video_path: str, out_fps: float) -> None:
    video_path = None
    for name in sorted(os.listdir(annotated_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            video_path = os.path.join(annotated_dir, name)
            break
    if video_path is None:
        raise RuntimeError(f"No annotated frames found in {annotated_dir}")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_video_path}")

    for name in sorted(os.listdir(annotated_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        image = cv2.imread(os.path.join(annotated_dir, name))
        if image is None:
            continue
        writer.write(image)
    writer.release()
    print(f"Annotated video saved to: {out_video_path}")


def annotate_video(
    video_path: str,
    config: str,
    weights: str,
    out_video_path: str,
    work_dir: str,
    draw_threshold: float,
    fps: Optional[float] = None,
    out_fps: Optional[float] = None,
) -> None:
    frames_dir = os.path.join(work_dir, "frames")
    annotated_dir = os.path.join(work_dir, "annotated")

    for stale_dir in (frames_dir, annotated_dir):
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir)

    print(f"Extracting frames from: {video_path}")
    split_video_to_frames(video_path, frames_dir, fps)

    print(f"Running inference with weights: {weights}")
    run_inference(config, weights, frames_dir, annotated_dir, draw_threshold)

    if out_fps is None:
        out_fps = fps if fps is not None else 30.0
    print("Assembling annotated video")
    annotated_frames_to_video(annotated_dir, out_video_path, out_fps)


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleDetection inference on a video and save the annotated video"
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--config", required=True, help="Path to the model config yml")
    parser.add_argument("--weights", required=True, help="Path to the trained weights (.pdparams or checkpoint dir)")
    parser.add_argument("--out-video", default=None, help="Path to the output annotated video (default: alongside input)")
    parser.add_argument("--work-dir", default=None, help="Directory for intermediate frames (default: temp dir, cleaned up)")
    parser.add_argument("--draw-threshold", type=float, default=0.2, help="Confidence threshold for drawing boxes")
    parser.add_argument("--fps", type=float, default=None, help="Sample frames at this rate instead of full video fps")
    parser.add_argument("--out-fps", type=float, default=None, help="Playback fps of the output video (default: same as --fps or 30)")
    args = parser.parse_args()

    if args.out_video is None:
        stem, ext = os.path.splitext(os.path.basename(args.video))
        args.out_video = os.path.join(os.path.dirname(args.video), f"{stem}_annotated{ext}")

    clean_up = args.work_dir is None
    work_dir = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate_work")
    os.makedirs(work_dir, exist_ok=True)

    try:
        annotate_video(
            args.video,
            args.config,
            args.weights,
            args.out_video,
            work_dir,
            args.draw_threshold,
            args.fps,
            args.out_fps,
        )
    finally:
        if clean_up:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
