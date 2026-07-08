# Phase-1 Cross-Domain D_link Diagnostic

- Mode: `smoke`
- Output directory: `results\phase1_crossdomain_dlink\20260507_120000_kth_smoke`
- Elapsed wall time: 8.8 s
- Seed: 0
- Tucker iterations: 180
- NTD-PL iterations: 180
- NTD-PL polynomial degree: 4
- Link refresh sample size: 400000

## Data Paths And Loader Notes

- KTH root=C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action; avi_files=599; annotated_segments=2391; units_selected=2391; processed_shape=24x72x96x3; rank=(12,18,18,3).

## Unit Definitions And Preprocessing

- BRDF: one MERL material file per unit; processed to 32x32x64x3, rank (12,12,16,2); official RGB scaling; invalid/negative values clamped; per-material max-normalization.
- Stanford LF: one decoded sub-aperture scene directory per unit when available; processed target 7x7x96x96x3, rank (4,4,16,16,2); no scene tensors are fabricated from calibration-only files.
- Harvard HS: one hyperspectral image per unit; ref/lbl .mat cubes processed to 128x128x31, rank (32,32,8); mask pixels zeroed when lbl exists; per-scene max-normalization.
- KTH-Action: one annotated action segment per unit; each clip is sampled to 24x72x96x3, rank (12,18,18,3); segments come from 00sequences.txt and are max-normalized independently.
- SAM is reported as NaN for BRDF because angular spectral error is not the target reflectance diagnostic here.

## Label Rule

Diagnostic labels are assigned only from D_link, not from NTD-PL gain: boundary is the bottom D_link tercile or D_link <= 0.01 dB; effective is the top tercile with D_link > 0.03 dB; all remaining successful units are moderate.

## Dataset Summary

| domain       | dataset    | num_units | num_ok | num_failed | median_d_link | mean_d_link | spearman_dlink_gain | mean_gain | median_gain | low_tercile_gain | mid_tercile_gain | high_tercile_gain | num_effective | num_moderate | num_boundary | processed_shape | rank         |
| ------------ | ---------- | --------- | ------ | ---------- | ------------- | ----------- | ------------------- | --------- | ----------- | ---------------- | ---------------- | ----------------- | ------------- | ------------ | ------------ | --------------- | ------------ |
| Action video | KTH-Action | 3         | 3      | 0          | 0.0421        | 0.0341      | 0.5000              | 2.6336    | 3.1500      | 1.1735           | 3.5773           | 3.1500            | 1             | 1            | 1            | 24x72x96x3      | (12,18,18,3) |

## Representative Units

| domain       | dataset    | unit_id                         | unit_label                    | selection             | diagnostic_label | d_link_db | tucker_rmse | ntdpl_rmse | rmse_gain_pct | rank         | processed_shape | source_path                                                                                      |
| ------------ | ---------- | ------------------------------- | ----------------------------- | --------------------- | ---------------- | --------- | ----------- | ---------- | ------------- | ------------ | --------------- | ------------------------------------------------------------------------------------------------ |
| Action video | KTH-Action | kth_person22_handclapping_d3_s2 | person22_handclapping_d3 seg2 | bottom-2 by D_link    | boundary         | 0.0052    | 0.0334      | 0.0330     | 1.1735        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\handclapping\person22_handclapping_d3_uncomp.avi |
| Action video | KTH-Action | kth_person16_walking_d3_s3      | person16_walking_d3 seg3      | bottom-2 by D_link    | moderate         | 0.0421    | 0.0529      | 0.0510     | 3.5773        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\walking\person16_walking_d3_uncomp.avi           |
| Action video | KTH-Action | kth_person16_walking_d3_s3      | person16_walking_d3 seg3      | top-2 by D_link       | moderate         | 0.0421    | 0.0529      | 0.0510     | 3.5773        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\walking\person16_walking_d3_uncomp.avi           |
| Action video | KTH-Action | kth_person13_running_d4_s1      | person13_running_d4 seg1      | top-2 by D_link       | effective        | 0.0549    | 0.0259      | 0.0251     | 3.1500        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\running\person13_running_d4_uncomp.avi           |
| Action video | KTH-Action | kth_person16_walking_d3_s3      | person16_walking_d3 seg3      | median-near by D_link | moderate         | 0.0421    | 0.0529      | 0.0510     | 3.5773        | (12,18,18,3) | 24x72x96x3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\walking\person16_walking_d3_uncomp.avi           |

## Failures

No failed fitted units. Some datasets may have no usable tensor units; see inspection notes.

## Diagnostic Reading

- BRDF/material reflectance: inspect median D_link, high-tercile gain, and representative materials before deciding whether it is a stronger link-yield domain.
- Light-field: usable only if decoded sub-aperture scene tensors are present; calibration-only data are reported as unavailable.
- Harvard HS: compare median D_link and high-tercile gain with RGB natural-image baselines before adding to Table 3.
- KTH-Action: treat the result as action-clip diagnosis rather than video classification; coherent temporal clips are the diagnostic unit.
- Table 3 recommendation: candidate evidence only; this script does not modify main.tex.

## Reproduction Command

```powershell
python scripts/run_phase1_crossdomain_dlink.py --mode smoke --datasets kth --brdf_root BRDFDatabase --lightfield_root caldata-B5143104560 --harvard_root CZ_hsdbi --kth_root KTH-Action --seed 0 --n_iter_max 180 --tucker_n_iter_max 180 --p_max 4 --jobs 2 --outdir results\phase1_crossdomain_dlink\20260507_120000_kth_smoke
```
