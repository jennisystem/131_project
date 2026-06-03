# End-to-end pipeline !
# load config, run full pipeline, return summary dict
import json
from pathlib import Path
import yaml
from tqdm import tqdm
from .detect import detect_ball_hsv, make_hsv_mask, clean_mask
from .fit import get_valid_points, fit_parabola_xy, ransac_parabola
from .metrics import tracking_summary, fit_summary
from .preprocess import prepare_frame
from .track import track_candidates
from .video_io import load_video_frames, save_video, save_vscode_video, save_detections_csv
from .visualize import annotate_frame, plot_trajectory, save_debug_frame


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

# load config, run full pipeline, return summary dict
def run_pipeline(
    config_path: str,
    video_path: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the pipeline. CLI overrides take precedence over config values.

    Parameters
    ----------
    config_path : path to YAML config.
    video_path  : optional override for video.input_path.
    output_dir  : optional override for video.output_dir. If not given but
                  video_path is, defaults to outputs/<video_stem>/.
    """
    cfg = load_config(config_path)

    video_cfg = cfg.get("video", {})

    if video_path is not None:
        input_path = video_path
        if output_dir is None:
            output_dir = f"outputs/{Path(video_path).stem}"
    else:
        input_path = video_cfg.get("input_path", "data/raw/sample_shot.mp4")
        if output_dir is None:
            output_dir = video_cfg.get("output_dir", "outputs/sample_shot")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    save_every = cfg.get("visualization", {}).get("save_debug_every_n_frames", 10)
    trail_length = cfg.get("tracking", {}).get("trail_length", 60)


    # load frames
    print(f"Loading frames from: {input_path}")
    frames, meta = load_video_frames(
        input_path,
        resize_width=video_cfg.get("resize_width"),
        start_frame=video_cfg.get("start_frame", 0),
        end_frame=video_cfg.get("end_frame"),
        frame_stride=video_cfg.get("frame_stride", 1),
    )
    fps = meta["fps"]
    print(f"  Loaded {len(frames)} frames  ({meta['resized_size'][0]}×{meta['resized_size'][1]})  @ {fps:.1f} fps")

    # per frame detection
    all_candidates: list[list[dict]] = []
    all_masks: list = []
    processed_frames: list = []

    hsv_cfg = cfg.get("hsv_threshold", {})
    morph_cfg = cfg.get("morphology", {})

    print("Running per-frame detection…")
    for frame in tqdm(frames, unit="frame"):
        proc = prepare_frame(frame, cfg)
        processed_frames.append(proc)

        mask = make_hsv_mask(proc, hsv_cfg.get("lower", [5, 80, 80]), hsv_cfg.get("upper", [25, 255, 255]))
        mask = clean_mask(mask, morph_cfg.get("open_kernel", 3), morph_cfg.get("close_kernel", 7))
        all_masks.append(mask)

        candidates = detect_ball_hsv(proc, cfg)
        all_candidates.append(candidates)

    # temporal tracking
    print("Tracking candidates across frames…")
    detections = track_candidates(all_candidates, cfg)
    for det, frame_idx in zip(detections, meta["frame_indices"]):
        det["frame_idx"] = frame_idx

    # save debug panels
    print(f"Saving debug panels (every {save_every} frames)…")
    for i, (frame_idx, frame, proc, mask, cands, det) in enumerate(
        zip(meta["frame_indices"], frames, processed_frames, all_masks, all_candidates, detections)
    ):
        if i % save_every == 0:
            save_debug_frame(
                frame_idx=frame_idx,
                original=frame,
                processed=proc,
                mask=mask,
                candidates=cands,
                detection=det,
                detections_so_far=detections[: i + 1],
                output_dir=str(debug_dir),
                trail_length=trail_length,
            )

    # fit trajectory
    print("Fitting trajectory…")
    fit_cfg = cfg.get("fitting", {})
    _, xs, ys = get_valid_points(detections)

    if fit_cfg.get("use_ransac", True) and len(xs) >= fit_cfg.get("min_inliers", 8):
        fit_result = ransac_parabola(
            xs, ys,
            iterations=fit_cfg.get("ransac_iterations", 300),
            threshold_px=fit_cfg.get("ransac_threshold_px", 20),
            min_inliers=fit_cfg.get("min_inliers", 8),
        )
    else:
        coeffs = fit_parabola_xy(xs, ys) if len(xs) >= 3 else None
        import numpy as np
        n = len(xs)
        fit_result = {
            "coeffs": coeffs,
            "inlier_mask": np.ones(n, dtype=bool),
            "outlier_mask": np.zeros(n, dtype=bool),
            "residuals": np.zeros(n),
            "success": coeffs is not None,
        }

    # annotated video   
    print("Building annotated video…")
    annotation_frames = (
        processed_frames
        if cfg.get("visualization", {}).get("use_processed_frames", False)
        else frames
    )
    annotated: list = []
    for i, (frame, det) in enumerate(zip(annotation_frames, detections)):
        ann = annotate_frame(frame, meta["frame_indices"][i], det, detections[: i + 1], trail_length)
        annotated.append(ann)
    annotated_path = output_dir / "annotated_video.mp4"
    vscode_path = output_dir / "annotated_video_h264.mp4"
    save_video(annotated, str(annotated_path), fps)
    if save_vscode_video(str(annotated_path), str(vscode_path)):
        print(f"Saved VS Code-friendly video: {vscode_path}")
    else:
        print("Skipping VS Code-friendly video: ffmpeg is not available.")

    # trajectory plot
    print("Saving trajectory plot…")
    plot_trajectory(detections, fit_result, str(output_dir / "trajectory_plot.png"))

    # save detections CSV
    save_detections_csv(detections, str(output_dir / "detections.csv"))

    # summary
    track_stats = tracking_summary(detections)
    fit_stats = fit_summary(fit_result)
    summary = {
        "video": input_path,
        **track_stats,
        **fit_stats,
        "notes": (
            "HSV thresholding detects the ball in clear sky/background frames "
            "but may struggle near the player's hands and during motion blur."
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        if k != "notes":
            print(f"  {k:<30s}: {v}")
    print(f"\n  Notes: {summary['notes']}")
    print("=" * 50)
    print(f"\nOutputs saved to: {output_dir}/")

    return summary
