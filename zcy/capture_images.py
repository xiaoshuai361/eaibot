#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import cv2


IMAGE_NAME_RE = re.compile(r"^image_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "captured_images"
CATEGORY_OUTPUT_DIRS = {
    1: SCRIPT_DIR / "照片" / "红绿灯",
    2: SCRIPT_DIR / "照片" / "楼宇",
    3: SCRIPT_DIR / "照片" / "人偶",
    4: SCRIPT_DIR / "照片" / "物块",
    5: SCRIPT_DIR / "照片" / "tag",
    6: SCRIPT_DIR / "照片" / "垃圾桶",
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture images from a camera. Press Space to save one frame."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for cv2.VideoCapture. Default 0.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save images. Default: captured_images next to this script.",
    )
    parser.add_argument(
        "--category",
        type=int,
        choices=sorted(CATEGORY_OUTPUT_DIRS.keys()),
        default=0,
        help="Optional category: 1=红绿灯, 2=楼宇, 3=人偶, 4=物块. Overrides --output-dir.",
    )
    parser.add_argument(
        "--ext",
        choices=("jpg", "png"),
        default="jpg",
        help="Image extension. Default jpg.",
    )
    parser.add_argument(
        "--resolution",
        type=positive_int,
        default=0,
        help="Optional square capture resolution, for example 320 means 320x320.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional capture width. Overrides --resolution when set. 0 means camera default.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional capture height. Overrides --resolution when set. 0 means camera default.",
    )
    args = parser.parse_args()
    if args.resolution:
        if args.width <= 0:
            args.width = args.resolution
        if args.height <= 0:
            args.height = args.resolution
    return args


def next_image_index(output_dir):
    max_index = 0
    if not output_dir.exists():
        return 1

    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = IMAGE_NAME_RE.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def open_camera(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera index {}. Try another --camera-index.".format(
                camera_index
            )
        )

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def main():
    args = parse_args()
    if args.category:
        output_dir = CATEGORY_OUTPUT_DIRS[args.category]
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    next_index = next_image_index(output_dir)
    cap = open_camera(args.camera_index, args.width, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Camera index: {}".format(args.camera_index))
    print("Capture size: {}x{}".format(actual_width, actual_height))
    print("Output dir: {}".format(output_dir))
    print("Next image: image_{}.{}".format(next_index, args.ext))
    print("Press Space to save, q or Esc to quit.")

    window_name = "capture_images"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame from camera.")
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                "Space: save  q/Esc: quit  next: image_{}.{}".format(
                    next_index, args.ext
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32:
                image_path = output_dir / "image_{}.{}".format(next_index, args.ext)
                if cv2.imwrite(str(image_path), frame):
                    print("Saved {}".format(image_path))
                    next_index += 1
                else:
                    print("Failed to save {}".format(image_path))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
