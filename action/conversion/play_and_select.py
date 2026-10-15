from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play ik.mp4 for demos in a list and select usable ones (y/n/r)."
    )
    parser.add_argument(
        "--list_file",
        type=str,
        default="/home/ycl/projects/workspace/spider/taco_usable_demos.txt",
        help="Input demo list (tab-separated: triplet<TAB>sequence).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/ycl/projects/datasets/egoengine",
        help="Dataset root containing TACO/Hand_Poses/...",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/ycl/projects/workspace/spider/rl_taco_usable_demos.txt",
        help="Output file in same tab-separated format.",
    )
    parser.add_argument(
        "--loop_video",
        action="store_true",
        help="Loop playback forever until you press q/esc in the video window.",
    )
    return parser.parse_args()


def load_demo_list(list_file: Path) -> list[tuple[str, str]]:
    demos: list[tuple[str, str]] = []
    with list_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            triplet, sequence = parts[0].strip(), parts[1].strip()
            if not triplet or not sequence:
                continue
            demos.append((triplet, sequence))
    return demos


def play_video(video_path: Path, window_name: str, loop_video: bool = False) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "n"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0
    delay_ms = max(int(1000.0 / fps), 1)

    choice = ""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # If video naturally ends and no decision yet, keep last frame and wait for key.
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0))
                ret2, frame2 = cap.read()
                if ret2:
                    cv2.imshow(window_name, frame2)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("y"), ord("Y")):
                    choice = "y"
                elif key in (ord("n"), ord("N")):
                    choice = "n"
                elif key in (ord("r"), ord("R")):
                    choice = "r"
                elif key in (27, ord("q"), ord("Q")):
                    choice = "n"
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("y"), ord("Y")):
                choice = "y"
                break
            if key in (ord("n"), ord("N")):
                choice = "n"
                break
            if key in (ord("r"), ord("R")):
                choice = "r"
                break
            if key in (27, ord("q"), ord("Q")):
                choice = "n"
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    return choice if choice in {"y", "n", "r"} else "n"


def write_output(output_file: Path, selected: list[tuple[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for triplet, sequence in selected:
            f.write(f"{triplet}\t{sequence}\n")


def main() -> None:
    args = parse_args()

    list_file = Path(args.list_file).expanduser()
    dataset_root = Path(args.dataset_root).expanduser()
    output_file = Path(args.output_file).expanduser()

    if not list_file.exists():
        raise FileNotFoundError(f"list_file not found: {list_file}")

    demos = load_demo_list(list_file)
    if not demos:
        print(f"No demos loaded from: {list_file}")
        write_output(output_file, [])
        return

    selected_usable: list[tuple[str, str]] = []

    print(f"Loaded {len(demos)} demos from {list_file}")
    print("Video controls during playback: y=usable, n=not usable, r=replay, q/esc=not usable.")

    for idx, (triplet, sequence) in enumerate(demos, start=1):
        video_path = (
            dataset_root
            / "TACO"
            / "Hand_Poses"
            / triplet
            / sequence
            / "ik.mp4"
        )

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(demos)}] {triplet} | {sequence}")
        print(f"Video: {video_path}")

        while True:
            if not video_path.exists():
                print("Video missing.")
                choice = "n"
            else:
                choice = play_video(
                    video_path,
                    window_name=f"{idx}/{len(demos)} {sequence}",
                    loop_video=args.loop_video,
                )
            if choice == "r":
                continue
            if choice == "y":
                selected_usable.append((triplet, sequence))
            break

        write_output(output_file, selected_usable)
        print(
            f"Current usable count: {len(selected_usable)} | "
            f"Saved progress to: {output_file}"
        )

    print("\nDone.")
    print(f"Final usable demos: {len(selected_usable)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
