# Phase-1 Cross-Domain D_link Diagnostic

- Mode: `smoke`
- Output directory: `results\phase1_crossdomain_dlink\20260507_123000_ucf101_smoke`
- Elapsed wall time: 8.5 s
- Seed: 0
- Tucker iterations: 180
- NTD-PL iterations: 180
- NTD-PL polynomial degree: 4
- Link refresh sample size: 400000

## Data Paths And Loader Notes

- UCF101 root=C:\Users\mrkys\workspace\NTD-PL\data\UCF101; split=train; csv_units=10055; units_selected=10055; processed_shape=24x72x96x3; rank=(12,18,18,3).

## Unit Definitions And Preprocessing

- BRDF: one MERL material file per unit; processed to 32x32x64x3, rank (12,12,16,2); official RGB scaling; invalid/negative values clamped; per-material max-normalization.
- Stanford LF: one decoded sub-aperture scene directory per unit when available; processed target 7x7x96x96x3, rank (4,4,16,16,2); no scene tensors are fabricated from calibration-only files.
- Harvard HS: one hyperspectral image per unit; ref/lbl .mat cubes processed to 128x128x31, rank (32,32,8); mask pixels zeroed when lbl exists; per-scene max-normalization.
- KTH-Action: one annotated action segment per unit; each clip is sampled to 24x72x96x3, rank (12,18,18,3); segments come from 00sequences.txt and are max-normalized independently.
- UCF101: one CSV-listed video clip per unit; each clip is sampled to 24x72x96x3, rank (12,18,18,3); clips are max-normalized independently.
- SAM is reported as NaN for BRDF because angular spectral error is not the target reflectance diagnostic here.

## Label Rule

Diagnostic labels are assigned only from D_link, not from NTD-PL gain: boundary is the bottom D_link tercile or D_link <= 0.01 dB; effective is the top tercile with D_link > 0.03 dB; all remaining successful units are moderate.

## Dataset Summary

| domain       | dataset | num_units | num_ok | num_failed | median_d_link | mean_d_link | spearman_dlink_gain | mean_gain | median_gain | low_tercile_gain | mid_tercile_gain | high_tercile_gain | num_effective | num_moderate | num_boundary | processed_shape | rank         |
| ------------ | ------- | --------- | ------ | ---------- | ------------- | ----------- | ------------------- | --------- | ----------- | ---------------- | ---------------- | ----------------- | ------------- | ------------ | ------------ | --------------- | ------------ |
| Action video | UCF101  | 3         | 3      | 0          | 0.0315        | 0.0590      | 1.0000              | 3.4963    | 0.9955      | 0.2576           | 0.9955           | 9.2359            | 1             | 1            | 1            | 24x72x96x3      | (12,18,18,3) |

## Representative Units

| domain       | dataset | unit_id                        | unit_label                            | selection             | diagnostic_label | d_link_db | tucker_rmse | ntdpl_rmse | rmse_gain_pct | rank         | processed_shape | source_path                                                                                 |
| ------------ | ------- | ------------------------------ | ------------------------------------- | --------------------- | ---------------- | --------- | ----------- | ---------- | ------------- | ------------ | --------------- | ------------------------------------------------------------------------------------------- |
| Action video | UCF101  | ucf101_v_LongJump_g08_c05      | LongJump:v_LongJump_g08_c05           | bottom-2 by D_link    | boundary         | 0.0063    | 0.0691      | 0.0689     | 0.2576        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\LongJump\v_LongJump_g08_c05.avi           |
| Action video | UCF101  | ucf101_v_StillRings_g05_c01    | StillRings:v_StillRings_g05_c01       | bottom-2 by D_link    | moderate         | 0.0315    | 0.0604      | 0.0598     | 0.9955        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\StillRings\v_StillRings_g05_c01.avi       |
| Action video | UCF101  | ucf101_v_StillRings_g05_c01    | StillRings:v_StillRings_g05_c01       | top-2 by D_link       | moderate         | 0.0315    | 0.0604      | 0.0598     | 0.9955        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\StillRings\v_StillRings_g05_c01.avi       |
| Action video | UCF101  | ucf101_v_PlayingGuitar_g18_c02 | PlayingGuitar:v_PlayingGuitar_g18_c02 | top-2 by D_link       | effective        | 0.1392    | 0.0485      | 0.0441     | 9.2359        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\PlayingGuitar\v_PlayingGuitar_g18_c02.avi |
| Action video | UCF101  | ucf101_v_StillRings_g05_c01    | StillRings:v_StillRings_g05_c01       | median-near by D_link | moderate         | 0.0315    | 0.0604      | 0.0598     | 0.9955        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\StillRings\v_StillRings_g05_c01.avi       |

## Failures

No failed fitted units. Some datasets may have no usable tensor units; see inspection notes.

## Diagnostic Reading

- BRDF/material reflectance: inspect median D_link, high-tercile gain, and representative materials before deciding whether it is a stronger link-yield domain.
- Light-field: usable only if decoded sub-aperture scene tensors are present; calibration-only data are reported as unavailable.
- Harvard HS: compare median D_link and high-tercile gain with RGB natural-image baselines before adding to Table 3.
- KTH-Action: treat the result as action-clip diagnosis rather than video classification; coherent temporal clips are the diagnostic unit.
- UCF101: treat the result as per-clip action-video diagnosis rather than classification; each clip is a self-contained tensor unit.
- Table 3 recommendation: candidate evidence only; this script does not modify main.tex.

## Reproduction Command

```powershell
python scripts/run_phase1_crossdomain_dlink.py --mode smoke --datasets ucf101 --brdf_root BRDFDatabase --lightfield_root caldata-B5143104560 --harvard_root CZ_hsdbi --kth_root KTH-Action --ucf101_root UCF101 --ucf101_split train --seed 0 --n_iter_max 180 --tucker_n_iter_max 180 --p_max 4 --jobs 2 --outdir results\phase1_crossdomain_dlink\20260507_123000_ucf101_smoke
```
