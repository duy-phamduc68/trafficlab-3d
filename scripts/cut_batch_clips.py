#!/usr/bin/env python3

import subprocess
import json
import os
import sys

def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8")

def probe_video(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of", "json",
        path
    ]
    data = json.loads(run(cmd))

    stream = data["streams"][0]
    fmt = data["format"]

    duration = float(fmt["duration"])

    fr = stream["avg_frame_rate"]
    if "/" in fr:
        num, den = fr.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fr)

    if stream.get("nb_frames"):
        frames = int(stream["nb_frames"])
    else:
        frames = int(round(duration * fps))

    return {
        "duration": duration,
        "frames": frames,
        "fps": fps,
        "width": stream["width"],
        "height": stream["height"]
    }

def print_info(title, info):
    print(title)
    print(f"  Duration   : {info['duration']:.6f} s")
    print(f"  Frames     : {info['frames']}")
    print(f"  Resolution : {info['width']}x{info['height']}")
    print(f"  FPS        : {info['fps']:.6f}")

def print_clip_info(n, base):
    info = {
        "duration": base["duration"] / n,
        "frames": int(round(base["frames"] / n)),
        "fps": base["fps"],
        "width": base["width"],
        "height": base["height"]
    }
    print_info(f"\n  Per-clip info if split into {n} parts:", info)

def cut_video(path, base, n):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = name
    os.makedirs(out_dir, exist_ok=True)

    clip_dur = base["duration"] / n
    fps = base["fps"]

    pad = max(3, len(str(n)))

    for i in range(n):
        start = clip_dur * i
        idx = str(i + 1).zfill(pad)
        out_path = os.path.join(out_dir, f"{name}_S{idx}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", path,
            "-ss", f"{start}",
            "-t", f"{clip_dur}",
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryslow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vsync", "cfr",
            "-r", f"{fps}",
            "-movflags", "+faststart",
            out_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"  Done → {out_dir}/")

def main():
    mp4s = [f for f in os.listdir(".") if f.lower().endswith(".mp4")]

    if not mp4s:
        print("No mp4 files found.")
        sys.exit(0)

    plan = []

    # -------- Phase 1: Planning --------
    for path in mp4s:
        print("\n" + "=" * 60)
        print(f"File: {path}")

        base = probe_video(path)
        print_info("Original video info:", base)

        chosen = None

        while True:
            user = input("\nEnter number of clips, or C to confirm: ").strip()

            if user.lower() == "c":
                if chosen is None:
                    print("No clip count chosen yet.")
                    continue
                plan.append((path, base, chosen))
                break

            try:
                n = int(user)
                if n <= 0:
                    raise ValueError
                chosen = n
                print_clip_info(n, base)
            except ValueError:
                print("Invalid input. Enter a positive integer or C.")

    # -------- Phase 2: Recap --------
    print("\n" + "=" * 60)
    print("Recap:")
    for path, _, n in plan:
        print(f"  {path} → {n} clips")

    confirm = input("\nProceed to cut ALL videos? (Y/N): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # -------- Phase 3: Execution --------
    print("\nStarting batch cut...\n")
    for path, base, n in plan:
        print(f"Cutting {path} into {n} clips")
        cut_video(path, base, n)

    print("\nAll videos processed.")

if __name__ == "__main__":
    main()
