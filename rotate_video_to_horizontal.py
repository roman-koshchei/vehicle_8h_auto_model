import argparse
import json
import os
import subprocess


def probe_video(video_path: str) -> tuple[int, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_side_data=rotation",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to probe video: {result.stderr}")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    rotation = 0
    for side_data in stream.get("side_data_list", []):
        if side_data.get("rotation") is not None:
            rotation = int(side_data["rotation"])
    return width, height, rotation


def rotate_video_to_horizontal(video_path: str, out_dir: str | None = None) -> str:
    width, height, rotation = probe_video(video_path)
    display_width, display_height = width, height
    if rotation in (90, -90, 270, -270):
        display_width, display_height = height, width

    if display_width >= display_height:
        print(f"Skip (already horizontal): {video_path}")
        return video_path

    stem, ext = os.path.splitext(os.path.basename(video_path))
    if out_dir is None:
        out_dir = os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_horizontal{ext}")

    transpose = "1"
    if rotation in (270, -90):
        transpose = "2"

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"transpose={transpose}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-metadata:s:v",
        "rotate=0",
        "-c:a",
        "copy",
        "-y",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to rotate video: {result.stderr}")
    print(f"Rotated: {video_path} -> {out_path}")
    return out_path


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
        description="Rotate portrait videos so width is greater than height, saving as a new file"
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
        "--out-dir",
        default=None,
        help="Directory to save rotated videos (default: same directory as each input)",
    )
    args = parser.parse_args()

    video_paths = list(args.video)
    if args.videos_dir is not None:
        video_paths.append(args.videos_dir)
    if not video_paths:
        parser.error("provide at least one of --video or --videos-dir")

    for video_path in resolve_video_paths(video_paths):
        rotate_video_to_horizontal(video_path, args.out_dir)


if __name__ == "__main__":
    main()