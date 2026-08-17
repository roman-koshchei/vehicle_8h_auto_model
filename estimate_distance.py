import argparse
import math


def estimate_distance(
    box_size_px: float,
    image_size_px: float,
    real_size_m: float,
    fov_deg: float,
) -> float:
    """Estimate distance to an object from its pixel size using the pinhole model.

    Assumptions:
      - Pinhole camera model with no lens distortion.
      - `fov_deg` is the field of view along the same image axis as `box_size_px`
        (horizontal FOV + box width, or vertical FOV + box height).
      - The object is roughly perpendicular to the optical axis (minimal yaw for
        width-based estimates), and its real-world size along that axis is known.
      - The image was not resized after capture; FOV matches the resolution used.

    Returns distance in meters.
    """
    if box_size_px <= 0 or image_size_px <= 0 or real_size_m <= 0 or fov_deg <= 0:
        raise ValueError("All inputs must be positive")
    if fov_deg >= 180:
        raise ValueError(f"FOV must be less than 180 degrees, got {fov_deg}")

    focal_length_px = (image_size_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    return real_size_m * focal_length_px / box_size_px


def estimate_distance_from_yolo_label(
    norm_width: float,
    norm_height: float,
    image_width_px: int,
    image_height_px: int,
    real_width_m: float,
    real_height_m: float,
    fov_horizontal_deg: float,
    fov_vertical_deg: float,
) -> dict[str, float | None]:
    """Estimate distance from a YOLO-format normalized box.

    Returns distances derived from the box width (horizontal FOV) and from the
    box height (vertical FOV). None is returned for axes that cannot be used.
    """
    box_width_px = norm_width * image_width_px
    box_height_px = norm_height * image_height_px

    distance_from_width = None
    if box_width_px > 0 and real_width_m > 0:
        distance_from_width = estimate_distance(
            box_width_px, image_width_px, real_width_m, fov_horizontal_deg
        )

    distance_from_height = None
    if box_height_px > 0 and real_height_m > 0:
        distance_from_height = estimate_distance(
            box_height_px, image_height_px, real_height_m, fov_vertical_deg
        )

    return {
        "distance_from_width_m": distance_from_width,
        "distance_from_height_m": distance_from_height,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate distance to an object from its pixel width using horizontal camera FOV"
    )
    parser.add_argument("--box-width-px", type=float, required=True, help="Object bounding box width in pixels")
    parser.add_argument("--image-width-px", type=float, required=True, help="Image width in pixels (horizontal axis)")
    parser.add_argument("--real-width-m", type=float, required=True, help="Real-world object width in meters")
    parser.add_argument("--fov-horizontal-deg", type=float, required=True, help="Camera horizontal FOV in degrees")
    args = parser.parse_args()

    distance_m = estimate_distance(
        args.box_width_px, args.image_width_px, args.real_width_m, args.fov_horizontal_deg
    )
    print(f"Estimated distance: {distance_m:.2f} m")


if __name__ == "__main__":
    main()