import argparse
import os
import subprocess


def split_video_to_frames(
    video_path: str, frames_directory_path: str, fps: float | None = None
):
    os.makedirs(frames_directory_path, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    filter_args = ["-vf", f"fps={fps}"] if fps is not None else []
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        *filter_args,
        "-y",
        os.path.join(frames_directory_path, f"{stem}_%04d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract frames: {result.stderr}")


video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def resolve_video_paths(paths: list[str]) -> list[str]:
    video_paths = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                file_path = os.path.join(path, name)
                if os.path.isfile(file_path) and os.path.splitext(name)[1].lower() in video_exts:
                    video_paths.append(file_path)
        else:
            video_paths.append(path)
    return video_paths


def main():
    parser = argparse.ArgumentParser(
        description="Split videos into frames at a given fps using ffmpeg"
    )
    parser.add_argument(
        "--video",
        nargs="+",
        default=[],
        help="Path to a video file or folder of videos (can be specified multiple times)",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Directory containing videos to process (equivalent to passing a folder via --video)",
    )
    parser.add_argument(
        "--frames-dir", required=True, help="Path to the output frames directory"
    )
    parser.add_argument("--fps", type=float, default=None, help="Frames per second (default: use video's native fps)")
    args = parser.parse_args()

    video_paths = list(args.video)
    if args.videos_dir is not None:
        video_paths.append(args.videos_dir)
    if not video_paths:
        parser.error("provide at least one of --video or --videos-dir")

    for video_path in resolve_video_paths(video_paths):
        split_video_to_frames(video_path, args.frames_dir, args.fps)


if __name__ == "__main__":
    main()