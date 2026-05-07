# Phase-1 Cross-Domain D_link Diagnostic

- Mode: `smoke`
- Output directory: `results\phase1_crossdomain_dlink\20260507_000000_smoke`
- Elapsed wall time: 10.1 s
- Seed: 0
- Tucker iterations: 180
- NTD-PL iterations: 180
- NTD-PL polynomial degree: 4
- Link refresh sample size: 400000

## Data Paths And Loader Notes

- BRDF root=C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase; binary_files=100; units_selected=100; processed_shape=32x32x64x3; rank=(12,12,16,2).
- No usable light-field scene tensors found under C:\Users\mrkys\workspace\NTD-PL\data\caldata-B5143104560; found 68 RAW and 68 metadata/calibration text files but no decoded sub-aperture images. Expected format: one scene directory containing a regular 7x7 or 9x9 grid of PNG/JPEG/TIFF sub-aperture views.
- Harvard root=C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi; cube_files=27; units_selected=27; processed_shape=128x128x31; rank=(32,32,8).

## Unit Definitions And Preprocessing

- BRDF: one MERL material file per unit; processed to 32x32x64x3, rank (12,12,16,2); official RGB scaling; invalid/negative values clamped; per-material max-normalization.
- Stanford LF: one decoded sub-aperture scene directory per unit when available; processed target 7x7x96x96x3, rank (4,4,16,16,2); no scene tensors are fabricated from calibration-only files.
- Harvard HS: one hyperspectral image per unit; ref/lbl .mat cubes processed to 128x128x31, rank (32,32,8); mask pixels zeroed when lbl exists; per-scene max-normalization.
- SAM is reported as NaN for BRDF because angular spectral error is not the target reflectance diagnostic here.

## Label Rule

Diagnostic labels are assigned only from D_link, not from NTD-PL gain: boundary is the bottom D_link tercile or D_link <= 0.01 dB; effective is the top tercile with D_link > 0.03 dB; all remaining successful units are moderate.

## Dataset Summary

| domain                  | dataset    | num_units | num_ok | num_failed | median_d_link | mean_d_link | spearman_dlink_gain | mean_gain | median_gain | low_tercile_gain | mid_tercile_gain | high_tercile_gain | num_effective | num_moderate | num_boundary | processed_shape | rank         |
| ----------------------- | ---------- | --------- | ------ | ---------- | ------------- | ----------- | ------------------- | --------- | ----------- | ---------------- | ---------------- | ----------------- | ------------- | ------------ | ------------ | --------------- | ------------ |
| Material reflectance    | BRDF       | 3         | 3      | 0          | 0.0261        | 0.3348      | 1.0000              | 7.7371    | 0.7331      | -1.1740          | 0.7331           | 23.6522           | 1             | 1            | 1            | 32x32x64x3      | (12,12,16,2) |
| Spectral natural scenes | Harvard HS | 3         | 3      | 0          | 0.0023        | 0.0026      | 0.5000              | -4.5461   | -4.1767     | -8.7517          | -0.7100          | -4.1767           | 0             | 0            | 3            | 128x128x31      | (32,32,8)    |

## Representative Units

| domain                  | dataset    | unit_id              | unit_label      | selection             | diagnostic_label | d_link_db | tucker_rmse | ntdpl_rmse | rmse_gain_pct | rank         | processed_shape | source_path                                                                    |
| ----------------------- | ---------- | -------------------- | --------------- | --------------------- | ---------------- | --------- | ----------- | ---------- | ------------- | ------------ | --------------- | ------------------------------------------------------------------------------ |
| Material reflectance    | BRDF       | brdf_pickled-oak-260 | pickled oak 260 | bottom-2 by D_link    | boundary         | 0.0013    | 0.0061      | 0.0061     | -1.1740       | (12,12,16,2) | 32x32x64x3      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pickled-oak-260.binary |
| Material reflectance    | BRDF       | brdf_red-fabric2     | red fabric2     | bottom-2 by D_link    | moderate         | 0.0261    | 0.0096      | 0.0095     | 0.7331        | (12,12,16,2) | 32x32x64x3      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric2.binary     |
| Material reflectance    | BRDF       | brdf_red-fabric2     | red fabric2     | top-2 by D_link       | moderate         | 0.0261    | 0.0096      | 0.0095     | 0.7331        | (12,12,16,2) | 32x32x64x3      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric2.binary     |
| Material reflectance    | BRDF       | brdf_steel           | steel           | top-2 by D_link       | effective        | 0.9771    | 0.0064      | 0.0049     | 23.6522       | (12,12,16,2) | 32x32x64x3      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\steel.binary           |
| Material reflectance    | BRDF       | brdf_red-fabric2     | red fabric2     | median-near by D_link | moderate         | 0.0261    | 0.0096      | 0.0095     | 0.7331        | (12,12,16,2) | 32x32x64x3      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric2.binary     |
| Spectral natural scenes | Harvard HS | harvard_imgh6        | imgh6           | bottom-2 by D_link    | boundary         | 0.0005    | 0.0043      | 0.0047     | -8.7517       | (32,32,8)    | 128x128x31      | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh6.mat                        |
| Spectral natural scenes | Harvard HS | harvard_img3         | img3            | bottom-2 by D_link    | boundary         | 0.0023    | 0.0088      | 0.0088     | -0.7100       | (32,32,8)    | 128x128x31      | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img3.mat                         |
| Spectral natural scenes | Harvard HS | harvard_img3         | img3            | top-2 by D_link       | boundary         | 0.0023    | 0.0088      | 0.0088     | -0.7100       | (32,32,8)    | 128x128x31      | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img3.mat                         |
| Spectral natural scenes | Harvard HS | harvard_img4         | img4            | top-2 by D_link       | boundary         | 0.0050    | 0.0042      | 0.0044     | -4.1767       | (32,32,8)    | 128x128x31      | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img4.mat                         |
| Spectral natural scenes | Harvard HS | harvard_img3         | img3            | median-near by D_link | boundary         | 0.0023    | 0.0088      | 0.0088     | -0.7100       | (32,32,8)    | 128x128x31      | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img3.mat                         |

## Failures

No failed fitted units. Some datasets may have no usable tensor units; see inspection notes.

## Diagnostic Reading

- BRDF/material reflectance: inspect median D_link, high-tercile gain, and representative materials before deciding whether it is a stronger link-yield domain.
- Light-field: usable only if decoded sub-aperture scene tensors are present; calibration-only data are reported as unavailable.
- Harvard HS: compare median D_link and high-tercile gain with RGB natural-image baselines before adding to Table 3.
- Table 3 recommendation: candidate evidence only; this script does not modify main.tex.

## Reproduction Command

```powershell
python scripts/run_phase1_crossdomain_dlink.py --mode smoke --datasets brdf,lightfield,harvard --brdf_root BRDFDatabase --lightfield_root caldata-B5143104560 --harvard_root CZ_hsdbi --seed 0 --n_iter_max 180 --tucker_n_iter_max 180 --p_max 4 --jobs 2 --outdir results\phase1_crossdomain_dlink\20260507_000000_smoke
```
