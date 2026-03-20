#!/usr/bin/env python3
"""
Combined script to create video grids from echo studies and compress them.
Processes echo DICOM data into grid videos with configurable resolution, fps, and duration.
"""

import os
import random
import cv2
import numpy as np
import shutil
import subprocess
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- Video Grid Creation Helpers -----------------
def load_media_handle(path):
    p = str(path).lower()
    if p.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        img = cv2.imread(str(path))
        if img is None:
            return None, 'bad'
        return {'kind':'image', 'frame': img}, 'ok'
    elif p.endswith('.mp4'):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, 'bad'
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {'kind':'video', 'cap': cap, 'fps': fps, 'frames': frames, 'w': w, 'h': h}, 'ok'
    else:
        return None, 'bad'

def close_media_handle(h):
    if h['kind'] == 'video':
        h['cap'].release()

def next_frame(handle, loop_mode='freeze_last'):
    """Return BGR frame or None if ended and not looping."""
    if handle['kind'] == 'image':
        return handle['frame']
    cap = handle['cap']
    ok, frame = cap.read()
    if ok:
        handle['_last'] = frame
        return frame
    # reached end
    if loop_mode == 'loop':
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if ok:
            handle['_last'] = frame
            return frame
    if '_last' in handle:
        return handle['_last'] if loop_mode == 'freeze_last' else None
    return None

def letterbox_to(frame, tw, th):
    h, w = frame.shape[:2]
    # keep aspect by letterboxing into target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    if (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    out[y0:y0+nh, x0:x0+nw] = frame
    return out

# ----------------- Grid Video Creation Worker -----------------
def _process_one_study(
    study_dir,
    output_dir,
    grid_size,
    tile_wh,
    fps_out,
    duration_sec,
    loop_mode,
    codec,
    seed
):
    """
    Runs in a separate process.
    Returns (study_name, out_path or None, message).
    """
    # try to avoid thread oversubscription in worker and suppress error messages
    try:
        cv2.setNumThreads(1)
        cv2.setLogLevel(0)  # Suppress OpenCV warnings/errors
    except Exception:
        pass
    # set RNG for reproducibility per worker if desired
    if seed is not None:
        random.seed(seed)

    study_dir = Path(study_dir)
    output_dir = Path(output_dir)

    tw, th = tile_wh
    coords = []
    for i in range(grid_size*grid_size):
        r, c = divmod(i, grid_size)
        y0, y1 = r*th, (r+1)*th
        x0, x1 = c*tw, (c+1)*tw
        coords.append((y0, y1, x0, x1))

    images_dir = study_dir / "images"
    if not images_dir.exists():
        return (study_dir.name, None, "No images directory")

    dicom_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith("dicom_")]
    need = grid_size * grid_size

    # Separate directories with videos vs. images only
    video_dirs = []
    image_dirs = []

    for d in dicom_dirs:
        if list(d.glob("*.mp4")):   # directory has at least one mp4
            video_dirs.append(d)
        else:
            image_dirs.append(d)

    # Start selecting from video-containing directories
    selected = []

    # First take as many video dirs as possible (randomly)
    if len(video_dirs) >= need:
        selected = random.sample(video_dirs, need)
    else:
        selected = video_dirs.copy()
        remaining = need - len(selected)
        if len(image_dirs) < remaining:
            return (study_dir.name, None, f"Only {len(dicom_dirs)} dicoms found, need {need}")
        selected.extend(random.sample(image_dirs, remaining))
    handles = []
    try:
        for d in selected:
            media_files = (list(d.glob("*.mp4")) + list(d.glob("*.jpg")) +
                           list(d.glob("*.jpeg")) + list(d.glob("*.png")))
            if not media_files:
                continue
            h, ok = load_media_handle(media_files[0])
            if ok != 'ok':
                continue
            handles.append(h)

        if len(handles) < need:
            for h in handles:
                close_media_handle(h)
            return (study_dir.name, None, f"Only loaded {len(handles)} media")

        grid_w, grid_h = grid_size*tw, grid_size*th
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out_path = output_dir / f"{study_dir.name}_grid.avi"  # MJPG works best as .avi
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (grid_w, grid_h))
        if not writer.isOpened():
            for h in handles:
                close_media_handle(h)
            return (study_dir.name, None, "Could not open VideoWriter")

        total_frames = int(fps_out * duration_sec)
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Precompute static tiles for images
        for h in handles:
            if h['kind'] == 'image':
                h['_tile'] = letterbox_to(h['frame'], tw, th)

        for _ in range(total_frames):
            for i, h in enumerate(handles):
                y0,y1,x0,x1 = coords[i]
                if h['kind'] == 'image':
                    tile = h['_tile']
                else:
                    f = next_frame(h, loop_mode=loop_mode)
                    if f is None:
                        tile = np.zeros((th, tw, 3), dtype=np.uint8)
                    else:
                        tile = letterbox_to(f, tw, th)
                grid[y0:y1, x0:x1] = tile
            writer.write(grid)

        writer.release()
        for h in handles:
            close_media_handle(h)

        return (study_dir.name, str(out_path), "OK")
    except Exception as e:
        # attempt to close resources on error
        try:
            for h in handles:
                close_media_handle(h)
        except Exception:
            pass
        return (study_dir.name, None, f"Error: {e}")

# ----------------- Video Compression Helpers -----------------
def _convert_with_ffmpeg(src: Path, dst: Path, target_resolution, target_fps):
    # letterbox to target resolution, cap to duration, target fps, MP4 H.264
    vf = f"scale={target_resolution}:{target_resolution}:force_original_aspect_ratio=decrease,pad={target_resolution}:{target_resolution}:(ow-iw)/2:(oh-ih)/2,fps={target_fps}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def _opencv_letterbox(frame, size):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    if (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = frame
    return out

def _try_open_writer_mp4(cv2, path_str, fps, wh):
    """
    Try a few MP4/H.264 fourccs. Return (writer, fourcc_used) or (None, None).
    Try codecs in order of preference. (Unmuted for reproducibility.)
    """
    # Try different fourcc codes in order of preference
    codecs_to_try = [
        "mp4v",  # Usually most compatible
        "avc1",  # H.264 variant
        "H264",  # Another H.264 variant
        "XVID",  # Fallback option
    ]
    
    for fourcc_tag in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            wr = cv2.VideoWriter(path_str, fourcc, fps, wh)
            if wr.isOpened():
                return wr, fourcc_tag
            wr.release()
        except Exception:
            # Don't try the next codec and return None
            print(f"Failed to open writer with codec {fourcc_tag}")
            return None, None
    return None, None

def _convert_with_opencv(src: Path, dst_mp4: Path, target_resolution, target_fps, duration_sec):
    """
    Try to write MP4 with OpenCV; if that fails, fall back to AVI (MJPG).
    Returns tuple (status, message, actual_output_path).
    """
    try:
        cv2.setNumThreads(1)
        # Suppress OpenCV error messages for codec trials
        cv2.setLogLevel(0)  # Set to silent mode
    except Exception:
        pass

    # Ensure out dir exists in worker
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return ("fail", f"cannot open input", None)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps_in / target_fps)))  # sample to target fps
    total_out = duration_sec * target_fps
    W = H = target_resolution

    # --- Try MP4 first ---
    writer, fourcc_used = _try_open_writer_mp4(cv2, str(dst_mp4), target_fps, (W, H))
    if writer is None:
        # --- Fallback to AVI (MJPG) ---
        dst_avi = dst_mp4.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(dst_avi), fourcc, target_fps, (W, H))
        if not writer.isOpened():
            cap.release()
            return ("fail", "cannot open any writer (no MP4 and no MJPG/AVI)", None)
        backend = f"opencv:MJPG->AVI ({dst_avi.name})"
        out_path = dst_avi
    else:
        backend = f"opencv:{fourcc_used}->MP4 ({dst_mp4.name})"
        out_path = dst_mp4

    frame_idx = 0
    written = 0
    ok_any = False
    while written < total_out:
        ok, frame = cap.read()
        if not ok:
            break
        ok_any = True
        if frame_idx % step == 0:
            tile = _opencv_letterbox(frame, target_resolution)
            writer.write(tile)
            written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    if not ok_any:
        # remove empty file if created
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass
        return ("fail", "no frames read", None)

    if written == 0:
        try:
            if out_path.exists() and out_path.stat().st_size == 0:
                out_path.unlink()
        except Exception:
            pass
        return ("fail", "wrote 0 frames", None)

    return ("ok", backend, out_path)

def _convert_one(src: str, dst_mp4: str, prefer_ffmpeg: bool, target_resolution: int, target_fps: int, duration_sec: int):
    src_p, dst_p = Path(src), Path(dst_mp4)
    try:
        if prefer_ffmpeg:
            _convert_with_ffmpeg(src_p, dst_p, target_resolution, target_fps)
            return (src_p.name, "ok", "ffmpeg->MP4", str(dst_p))
        else:
            status, msg, outp = _convert_with_opencv(src_p, dst_p, target_resolution, target_fps, duration_sec)
            return (src_p.name, status, msg, str(outp) if outp else None)
    except subprocess.CalledProcessError:
        return (src_p.name, "fail", "ffmpeg failed", None)
    except Exception as e:
        return (src_p.name, "fail", f"error: {e}", None)

# ----------------- Combined Processing Worker -----------------
def _process_study_with_compression(
    study_dir,
    grid_output_dir,
    compressed_output_dir,
    grid_size,
    tile_wh,
    fps_out,
    duration_sec,
    loop_mode,
    codec,
    seed,
    # Compression params
    target_resolution,
    target_fps,
    compress_duration_sec,
    prefer_ffmpeg,
    skip_compression=False
):
    """
    Process one study: create grid video, compress it, then clean up intermediate file.
    Returns (study_name, final_path or None, message).
    """
    study_dir = Path(study_dir)
    study_name = study_dir.name
    
    # Step 1: Create grid video
    grid_result = _process_one_study(
        str(study_dir),
        str(grid_output_dir),
        grid_size,
        tile_wh,
        fps_out,
        duration_sec,
        loop_mode,
        codec,
        seed
    )
    
    study_name_result, grid_path, grid_msg = grid_result
    
    if grid_path is None:
        return (study_name, None, f"Grid creation failed: {grid_msg}")
    
    if skip_compression:
        return (study_name, grid_path, "Grid video created (compression skipped)")
    
    # Step 2: Compress the grid video
    grid_path_obj = Path(grid_path)
    compressed_path = Path(compressed_output_dir) / f"{grid_path_obj.stem}_small.mp4"
    
    try:
        # Check if compressed version already exists
        if compressed_path.exists():
            # Clean up the intermediate grid file
            try:
                grid_path_obj.unlink()
            except Exception:
                pass
            return (study_name, str(compressed_path), "Compressed video already exists")
        
        # Compress the video
        compress_result = _convert_one(
            str(grid_path_obj),
            str(compressed_path),
            prefer_ffmpeg,
            target_resolution,
            target_fps,
            compress_duration_sec
        )
        
        _, status, msg, final_path = compress_result
        
        # Step 3: Clean up intermediate grid file if compression was successful
        if status == "ok":
            try:
                grid_path_obj.unlink()
                return (study_name, final_path, f"Compressed and cleaned up: {msg}")
            except Exception as e:
                return (study_name, final_path, f"Compressed but cleanup failed: {msg} (cleanup error: {e})")
        else:
            return (study_name, None, f"Compression failed: {msg}")
            
    except Exception as e:
        return (study_name, None, f"Error during compression: {e}")

# ----------------- Main Processing Functions -----------------
def process_studies_with_immediate_cleanup(
    echo_dir,
    grid_output_dir,
    compressed_output_dir,
    grid_size=5,
    tile_wh=(256,256),
    fps_out=24,
    duration_sec=10,
    loop_mode='freeze_last',
    codec='MJPG',
    max_workers=20,
    max_studies=None,
    deterministic=False,
    # Compression params
    target_resolution=700,
    target_fps=5,
    compress_duration_sec=5,
    skip_compression=False
):
    """
    Process studies with immediate compression and cleanup after each video.
    """
    echo_dir = Path(echo_dir)
    grid_output_dir = Path(grid_output_dir)
    compressed_output_dir = Path(compressed_output_dir)
    
    grid_output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_compression:
        compressed_output_dir.mkdir(parents=True, exist_ok=True)

    # choose studies
    study_dirs = [d for d in echo_dir.iterdir() if d.is_dir()]
    if max_studies is not None:
        study_dirs = study_dirs[:max_studies]

    workers = max(1, min(int(max_workers), (os.cpu_count() or 1)))

    print(f"Processing {len(study_dirs)} studies with {workers} workers (immediate cleanup)...")
    
    # Setup compression preferences
    prefer_ffmpeg = shutil.which("ffmpeg") is not None
    if prefer_ffmpeg:
        # keep external libs from oversubscribing CPUs
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    futures, results = [], []

    def seed_for(study_name):
        return (hash(study_name) & 0xFFFFFFFF) if deterministic else None

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sd in study_dirs:
            # Check if final output already exists
            if skip_compression:
                expected_output = grid_output_dir / f"{sd.name}_grid.avi"
            else:
                expected_output = compressed_output_dir / f"{sd.name}_grid_small.mp4"
                # Also check for .avi fallback
                expected_output_avi = compressed_output_dir / f"{sd.name}_grid_small.avi"
                if expected_output.exists() or expected_output_avi.exists():
                    print(f"[{sd.name}] Skipping (output already exists)")
                    results.append((sd.name, str(expected_output if expected_output.exists() else expected_output_avi), "Already exists"))
                    continue
            
            if expected_output.exists():
                print(f"[{sd.name}] Skipping (output already exists)")
                results.append((sd.name, str(expected_output), "Already exists"))
                continue

            futures.append(
                ex.submit(
                    _process_study_with_compression,
                    str(sd),
                    str(grid_output_dir),
                    str(compressed_output_dir),
                    grid_size,
                    tile_wh,
                    fps_out,
                    duration_sec,
                    loop_mode,
                    codec,
                    seed_for(sd.name),
                    target_resolution,
                    target_fps,
                    compress_duration_sec,
                    prefer_ffmpeg,
                    skip_compression
                )
            )

        for f in as_completed(futures):
            study_name, final_path, msg = f.result()
            if final_path:
                print(f"[{study_name}] ✓ {msg}")
            else:
                print(f"[{study_name}] ✗ {msg}")
            results.append((study_name, final_path, msg))

    return results



def main():
    parser = argparse.ArgumentParser(description="Create and compress echo video grids")
    parser.add_argument("--echo-dir", default="/raid/projects/masadi/echo_extracted_new_test", 
                       help="Directory containing echo studies")
    parser.add_argument("--output-dir", default="./echo_video_grids_new_test",
                       help="Output directory for processed videos")
    parser.add_argument("--resolution", type=int, default=700,
                       help="Target resolution for compressed videos (default: 700)")
    parser.add_argument("--fps", type=int, default=5,
                       help="Target FPS for compressed videos (default: 5)")
    parser.add_argument("--duration", type=int, default=5,
                       help="Duration in seconds for compressed videos (default: 5)")
    parser.add_argument("--grid-size", type=int, default=5,
                       help="Grid size (NxN) for video grid (default: 5)")
    parser.add_argument("--max-studies", type=int, default=None,
                       help="Maximum number of studies to process (default: 10)")
    parser.add_argument("--no-compress", action="store_true",
                       help="Skip compression step")
    parser.add_argument("--max-cores", type=int, default=100,
                       help="Maximum number of CPU cores to use for parallel processing (default: 50)")
    
    args = parser.parse_args()
    
    echo_dir = Path(args.echo_dir)
    output_dir = Path(args.output_dir)
    
    if not echo_dir.exists():
        print(f"Error: Echo directory {echo_dir} does not exist")
        return 1
    
    # Create intermediate and final output directories
    grid_dir = output_dir / "grid_videos"
    compressed_dir = output_dir / "compressed_videos"
    
    # Calculate worker distribution based on user-specified max cores
    total_cpu_limit = args.max_cores
    available_cpus = min(os.cpu_count() or 1, total_cpu_limit)
    
    # Use 60% of CPUs for grid creation, 40% for compression
    grid_workers = max(1, int(available_cpus * 0.6))
    compress_workers = max(1, int(available_cpus * 0.4))
    
    print(f"Processing up to {args.max_studies} studies")
    print(f"Using max {total_cpu_limit} cores (immediate cleanup after each video)")
    print(f"Output resolution: {args.resolution}x{args.resolution}, FPS: {args.fps}, Duration: {args.duration}s")
    
    # Process studies with immediate compression and cleanup
    tile_size = 256  # Fixed tile size for grid creation
    results = process_studies_with_immediate_cleanup(
        echo_dir=echo_dir,
        grid_output_dir=grid_dir,
        compressed_output_dir=compressed_dir,
        grid_size=args.grid_size,
        tile_wh=(tile_size, tile_size),
        fps_out=24,  # High FPS for grid creation, will be reduced in compression
        duration_sec=args.duration,
        max_workers=total_cpu_limit,  # Use all available cores for the combined process
        max_studies=args.max_studies,
        target_resolution=args.resolution,
        target_fps=args.fps,
        compress_duration_sec=args.duration,
        skip_compression=args.no_compress
    )
    
    # Clean up empty grid directory if compression was used
    if not args.no_compress:
        try:
            if grid_dir.exists() and not any(grid_dir.iterdir()):
                grid_dir.rmdir()
        except Exception:
            pass  # Ignore if directory is not empty or can't be removed
    
    # Summary
    print("\n=== Summary ===")
    success_count = sum(1 for _, path, _ in results if path is not None)
    total_count = len(results)
    
    print(f"Successfully processed: {success_count}/{total_count} studies")
    
    if args.no_compress:
        print(f"Raw grid videos location: {grid_dir}")
    else:
        print(f"Final compressed videos location: {compressed_dir}")
        print("(Intermediate grid videos were cleaned up immediately after each compression)")
    
    return 0

if __name__ == "__main__":
    exit(main())
