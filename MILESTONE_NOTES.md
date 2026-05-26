# Milestone Check-In Notes

## Technical Progress

We implemented an initial classical computer vision pipeline for tracking a
basketball shot from a side-view phone video. The current system loads a video,
extracts frames, optionally crops and smooths each frame, applies HSV color
thresholding to isolate orange basketball-like pixels, cleans the mask using
morphological operations, and extracts contour-based ball candidates. Candidate
regions are filtered by area and circularity, then linked across frames using a
simple motion-continuity tracker. We also implemented preliminary parabolic
trajectory fitting using least squares and RANSAC. Since the initial version,
we tested the pipeline on two additional sample shots to see how well the same
thresholding, filtering, and tracking steps generalize across different shot
examples.

## Visualization and Results

The current debug outputs include side-by-side panels showing the original
frame, HSV mask, contour candidates, and selected ball detection. These
visualizations help show where the pipeline works and where it fails. In early
tests across the original video and two additional sample shots, HSV
thresholding is most successful when the ball is separated from the player's
body and background. The main failure cases are motion blur, occlusion near the
player's hands, and false positives from similarly colored objects such as the
rim or background. The trajectory plot shows raw tracked centers and a fitted
parabolic arc, giving an initial visualization of the shot path.

## Updated Timeline and Plan

For the next stage, we will tune HSV thresholds and contour filters across
additional videos, improve temporal tracking with stronger motion and radius
consistency checks, and manually label a subset of frames to compute center
error and tracking success rate. If Hough circle detection is not stable, we
will keep it as a comparison method rather than the main pipeline. The final
project will focus on a reliable 2D tracking system with annotated video,
trajectory plots, quantitative evaluation, and a clear discussion of
limitations.

---

## Completed / In Progress

- Implemented frame extraction and preprocessing
- Implemented HSV threshold segmentation
- Implemented contour-based candidate extraction
- Generated debug visualizations (2×2 panels + gallery)
- Implemented simple constant-velocity temporal tracking
- Implemented least-squares + RANSAC parabola fitting
- Annotated video output working
- Trajectory plot (inverted y-axis, inlier/outlier coded) working
- Quantitative summary JSON working
- Tested the pipeline on two additional sample shots

## Next Steps

- Tune HSV thresholds and ROI on multiple videos
- Add radius consistency check in tracker
- Manually label a subset of frames for quantitative evaluation
- Implement optional Canny/Hough circle comparison if HSV is not reliable enough
- Prepare final visualizations and discussion of failure cases

## Pivot Note

If Hough circle detection is noisy or unreliable, keep it as a comparison
only and focus the final method on HSV segmentation + contour filtering +
temporal tracking + RANSAC. This is more feasible and still aligned with the
original proposal.
