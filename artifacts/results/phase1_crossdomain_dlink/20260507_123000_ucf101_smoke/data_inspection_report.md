# Phase-1 Cross-Domain D_link Data Inspection

- Output directory: `results\phase1_crossdomain_dlink\20260507_123000_ucf101_smoke`
- Datasets requested: `ucf101`

## File Types

| root   | extension | count |
| ------ | --------- | ----- |
| UCF101 | .avi      | 13451 |
| UCF101 | .csv      | 3     |

## Loader And Unit Notes

- UCF101 root=C:\Users\mrkys\workspace\NTD-PL\data\UCF101; split=train; csv_units=10055; units_selected=10055; processed_shape=24x72x96x3; rank=(12,18,18,3).
- BRDF loader: minimal MERL binary reader using dimensions header and official RGB scales.
- Harvard loader: MATLAB reader chooses a 3D numeric cube, preferring ref/reflectances/rad/img/hypercube/cube.
- Light-field loader: requires decoded sub-aperture image grids; raw calibration-only Lytro unit data are not treated as scene tensors.
- KTH loader: reads AVI clips with imageio-ffmpeg, then crops to annotated action segments from 00sequences.txt.
- UCF101 loader: reads AVI clips with imageio-ffmpeg and treats each CSV-listed video clip as one unit.

## Candidate Units

| domain       | dataset | unit_id                        | source_path                                                                                 | raw_shape | processed_shape | rank         |
| ------------ | ------- | ------------------------------ | ------------------------------------------------------------------------------------------- | --------- | --------------- | ------------ |
| Action video | UCF101  | ucf101_v_LongJump_g08_c05      | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\LongJump\v_LongJump_g08_c05.avi           |           | 24x72x96x3      | (12,18,18,3) |
| Action video | UCF101  | ucf101_v_PlayingGuitar_g18_c02 | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\PlayingGuitar\v_PlayingGuitar_g18_c02.avi |           | 24x72x96x3      | (12,18,18,3) |
| Action video | UCF101  | ucf101_v_StillRings_g05_c01    | C:\Users\mrkys\workspace\NTD-PL\data\UCF101\train\StillRings\v_StillRings_g05_c01.avi       |           | 24x72x96x3      | (12,18,18,3) |
