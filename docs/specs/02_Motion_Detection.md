# Motion Detection (Background Subtraction)


## Algorithm

## Output

Notes (VI): Phát hiện chuyển động bằng nền MOG2, ngưỡng pixel > 150.

## Component Purpose
- Gate heavier detectors (pose/fall/seizure) only when there is meaningful scene activity.
- Provide a stable, shadow-free motion mask resilient to noise and minor lighting changes.

Giải thích (VI): Dùng phát hiện chuyển động để "mở cổng" cho bước AI nặng, giảm tải và nhiễu.

## Inputs & Configuration
- Input: BGR frame (`np.ndarray`) from camera ingestion.
- Config parameters:
	- `resolution=(256,144)` downscale target.
	- `history=200` frames for MOG2 background model.
	- `varThreshold=32` sensitivity to pixel variance.
	- `detectShadows=True` to mark shadows as 127 (to be removed).
	- `kernel=3×3 ellipse` for morphological opening.
	- `threshold=150` white pixels to flag motion.
	- `start_frames` warmup count before decisions.

## Processing Steps (Runtime Flow)
1. Receive frame → downscale to `resolution` to reduce compute and stabilize noise.
2. Apply `cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)` → get `mask`.
3. Shadow removal: set `mask[mask==127] = 0` to ignore shadow-labeled pixels.
4. Morphological opening: `cv2.morphologyEx(mask, MORPH_OPEN, ellipse_3x3)` to remove speckles.
5. Count motion pixels: `motion_pixels = cv2.countNonZero(mask)`.
6. Warmup guard: if `frame_count < start_frames` → `motion_detected=False` (model still stabilizing).
7. Decision: `motion_detected = motion_pixels > threshold`.
8. Emit `{motion_detected, motion_pixels, threshold, frame_count}` and optionally the cleaned `mask` for downstream visualization.

Giải thích (VI): Luồng xử lý gồm thu nhỏ ảnh, trừ nền, bỏ bóng, lọc nhiễu, đếm pixel trắng rồi so ngưỡng sau giai đoạn khởi động.

## State Machine
- States: `WARMING_UP` → `READY`.
- Transitions:
	- `WARMING_UP→READY`: when `frame_count ≥ start_frames`.
	- Decision occurs only in `READY` to avoid false positives during background learning.

## Parameters & Defaults
- `resolution`: `(256,144)` → balance speed and sensitivity; larger values increase sensitivity to small motions.
- `history`: `200` → longer history stabilizes background but slows adaptation.
- `varThreshold`: `32` → lower values are more sensitive; increase to reduce flicker.
- `detectShadows`: `True` → enables shadow labeling; must remove 127 to avoid shadow-triggered motion.
- `kernel`: `ellipse(3×3)` → smooths mask; can increase to suppress high-frequency noise.
- `threshold`: `150` pixels → environment-dependent; scale proportional to resolution.
- `start_frames`: e.g., `30–120` depending on camera dynamics.

## Metrics & Telemetry
- `motion_pixels` per frame (primary metric).
- `frame_count` since start/reset.
- Optional: `mask_coverage = motion_pixels / (H×W)` to normalize across resolutions.

## Edge Cases & Filters
- Lighting flicker/AE changes: raise `varThreshold`, increase `history`, or add temporal median of `motion_pixels`.
- Shadows: ensure `detectShadows=True` and zero-out `127` labels.
- Camera noise: apply stronger blur before MOG2 or larger opening kernel.
- Static small movements (fans, leaves): raise `threshold` or expand kernel.

## Pseudocode (Reference)
```
state = WARMING_UP
frame_count = 0
bg = MOG2(history=200, varThreshold=32, detectShadows=True)
for frame in stream:
		small = resize(frame, (256,144))
		mask = bg.apply(small)
		mask[mask == 127] = 0
		mask = morphology_open(mask, ellipse3x3)
		motion_pixels = count_non_zero(mask)
		frame_count += 1
		if frame_count < start_frames:
				motion_detected = False
				continue
		state = READY
		motion_detected = motion_pixels > threshold
		yield {motion_detected, motion_pixels, threshold, frame_count}
```

## Testing Checklist
- Warmup behavior: verify no motion flagged during `start_frames`.
- Sensitivity: confirm small motions exceed `threshold` while noise does not.
- Robustness: test under shadow changes and light flicker; adjust `varThreshold/history`.

## Tuning Notes
- Scale `threshold` with resolution: doubling dimensions roughly quadruples pixel area.
- For noisy cameras, increase kernel size or apply Gaussian blur pre-MOG2.
- For fast adaptation (busy scenes), lower `history`; for stability (static scenes), increase it.

Notes (VI): Điều chỉnh ngưỡng theo độ phân giải, tăng `history` cho nền ổn định, và kiểm thử warmup để tránh báo giả.
