# Phase-1 Cross-Domain D_link Data Inspection

- Output directory: `results\phase1_crossdomain_dlink\20260507_000000_smoke`
- Datasets requested: `brdf,lightfield,harvard`

## File Types

| root                | extension | count |
| ------------------- | --------- | ----- |
| BRDFDatabase        | .binary   | 100   |
| BRDFDatabase        | .cpp      | 1     |
| BRDFDatabase        | .md       | 1     |
| BRDFDatabase        | .txt      | 1     |
| caldata-B5143104560 | .GCT      | 34    |
| caldata-B5143104560 | .HST      | 34    |
| caldata-B5143104560 | .RAW      | 34    |
| caldata-B5143104560 | .TXT      | 34    |
| caldata-B5143104560 | .json     | 7     |
| caldata-B5143104560 | .BIN      | 1     |
| CZ_hsdbi            | .mat      | 27    |
| CZ_hsdbi            | .txt      | 2     |

## Loader And Unit Notes

- BRDF root=C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase; binary_files=100; units_selected=100; processed_shape=32x32x64x3; rank=(12,12,16,2).
- No usable light-field scene tensors found under C:\Users\mrkys\workspace\NTD-PL\data\caldata-B5143104560; found 68 RAW and 68 metadata/calibration text files but no decoded sub-aperture images. Expected format: one scene directory containing a regular 7x7 or 9x9 grid of PNG/JPEG/TIFF sub-aperture views.
- Harvard root=C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi; cube_files=27; units_selected=27; processed_shape=128x128x31; rank=(32,32,8).
- BRDF loader: minimal MERL binary reader using dimensions header and official RGB scales.
- Harvard loader: MATLAB reader chooses a 3D numeric cube, preferring ref/reflectances/rad/img/hypercube/cube.
- Light-field loader: requires decoded sub-aperture image grids; raw calibration-only Lytro unit data are not treated as scene tensors.

## Candidate Units

| domain                  | dataset    | unit_id              | source_path                                                                    | raw_shape   | processed_shape | rank         |
| ----------------------- | ---------- | -------------------- | ------------------------------------------------------------------------------ | ----------- | --------------- | ------------ |
| Material reflectance    | BRDF       | brdf_pickled-oak-260 | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pickled-oak-260.binary | 90x90x180x3 | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-fabric2     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric2.binary     | 90x90x180x3 | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_steel           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\steel.binary           | 90x90x180x3 | 32x32x64x3      | (12,12,16,2) |
| Spectral natural scenes | Harvard HS | harvard_img3         | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img3.mat                         |             | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_img4         | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img4.mat                         |             | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgh6        | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh6.mat                        |             | 128x128x31      | (32,32,8)    |
