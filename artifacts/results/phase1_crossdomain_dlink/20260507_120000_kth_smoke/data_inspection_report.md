# Phase-1 Cross-Domain D_link Data Inspection

- Output directory: `results\phase1_crossdomain_dlink\20260507_120000_kth_smoke`
- Datasets requested: `kth`

## File Types

| root       | extension | count |
| ---------- | --------- | ----- |
| KTH-Action | .avi      | 599   |
| KTH-Action | .txt      | 1     |

## Loader And Unit Notes

- KTH root=C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action; avi_files=599; annotated_segments=2391; units_selected=2391; processed_shape=24x72x96x3; rank=(12,18,18,3).
- BRDF loader: minimal MERL binary reader using dimensions header and official RGB scales.
- Harvard loader: MATLAB reader chooses a 3D numeric cube, preferring ref/reflectances/rad/img/hypercube/cube.
- Light-field loader: requires decoded sub-aperture image grids; raw calibration-only Lytro unit data are not treated as scene tensors.
- KTH loader: reads AVI clips with imageio-ffmpeg, then crops to annotated action segments from 00sequences.txt.

## Candidate Units

| domain       | dataset    | unit_id                         | source_path                                                                                      | raw_shape | processed_shape | rank         |
| ------------ | ---------- | ------------------------------- | ------------------------------------------------------------------------------------------------ | --------- | --------------- | ------------ |
| Action video | KTH-Action | kth_person13_running_d4_s1      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\running\person13_running_d4_uncomp.avi           |           | 24x72x96x3      | (12,18,18,3) |
| Action video | KTH-Action | kth_person16_walking_d3_s3      | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\walking\person16_walking_d3_uncomp.avi           |           | 24x72x96x3      | (12,18,18,3) |
| Action video | KTH-Action | kth_person22_handclapping_d3_s2 | C:\Users\mrkys\workspace\NTD-PL\data\KTH-Action\handclapping\person22_handclapping_d3_uncomp.avi |           | 24x72x96x3      | (12,18,18,3) |
