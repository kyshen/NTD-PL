# Phase-1 Cross-Domain D_link Data Inspection

- Output directory: `results\phase1_crossdomain_dlink\20260507_000000_full`
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
- No usable light-field scene tensors found under C:\Users\mrkys\workspace\NTD-PL\data\caldata-B5143104560; found 34 RAW and 34 metadata/calibration text files but no decoded sub-aperture images. Expected format: one scene directory containing a regular 7x7 or 9x9 grid of PNG/JPEG/TIFF sub-aperture views.
- Harvard root=C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi; cube_files=27; units_selected=27; processed_shape=128x128x31; rank=(32,32,8).
- BRDF loader: minimal MERL binary reader using dimensions header and official RGB scales.
- Harvard loader: MATLAB reader chooses a 3D numeric cube, preferring ref/reflectances/rad/img/hypercube/cube.
- Light-field loader: requires decoded sub-aperture image grids; raw calibration-only Lytro unit data are not treated as scene tensors.

## Candidate Units

| domain                  | dataset    | unit_id                       | source_path                                                                             | raw_shape    | processed_shape | rank         |
| ----------------------- | ---------- | ----------------------------- | --------------------------------------------------------------------------------------- | ------------ | --------------- | ------------ |
| Material reflectance    | BRDF       | brdf_alum-bronze              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\alum-bronze.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_alumina-oxide            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\alumina-oxide.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_aluminium                | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\aluminium.binary                | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_aventurnine              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\aventurnine.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_beige-fabric             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\beige-fabric.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_black-fabric             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\black-fabric.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_black-obsidian           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\black-obsidian.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_black-oxidized-steel     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\black-oxidized-steel.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_black-phenolic           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\black-phenolic.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_black-soft-plastic       | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\black-soft-plastic.binary       | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_blue-acrylic             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\blue-acrylic.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_blue-fabric              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\blue-fabric.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_blue-metallic-paint      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\blue-metallic-paint.binary      | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_blue-metallic-paint2     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\blue-metallic-paint2.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_blue-rubber              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\blue-rubber.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_brass                    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\brass.binary                    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_cherry-235               | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\cherry-235.binary               | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_chrome-steel             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\chrome-steel.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_chrome                   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\chrome.binary                   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_colonial-maple-223       | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\colonial-maple-223.binary       | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_color-changing-paint1    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\color-changing-paint1.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_color-changing-paint2    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\color-changing-paint2.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_color-changing-paint3    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\color-changing-paint3.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_dark-blue-paint          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\dark-blue-paint.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_dark-red-paint           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\dark-red-paint.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_dark-specular-fabric     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\dark-specular-fabric.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_delrin                   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\delrin.binary                   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_fruitwood-241            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\fruitwood-241.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_gold-metallic-paint      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\gold-metallic-paint.binary      | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_gold-metallic-paint2     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\gold-metallic-paint2.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_gold-metallic-paint3     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\gold-metallic-paint3.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_gold-paint               | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\gold-paint.binary               | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_gray-plastic             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\gray-plastic.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_grease-covered-steel     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\grease-covered-steel.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-acrylic            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-acrylic.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-fabric             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-fabric.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-latex              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-latex.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-metallic-paint     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-metallic-paint.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-metallic-paint2    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-metallic-paint2.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_green-plastic            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\green-plastic.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_hematite                 | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\hematite.binary                 | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_ipswich-pine-221         | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\ipswich-pine-221.binary         | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_light-brown-fabric       | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\light-brown-fabric.binary       | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_light-red-paint          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\light-red-paint.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_maroon-plastic           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\maroon-plastic.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_natural-209              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\natural-209.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_neoprene-rubber          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\neoprene-rubber.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_nickel                   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\nickel.binary                   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_nylon                    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\nylon.binary                    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_orange-paint             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\orange-paint.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pearl-paint              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pearl-paint.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pickled-oak-260          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pickled-oak-260.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pink-fabric              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pink-fabric.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pink-fabric2             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pink-fabric2.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pink-felt                | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pink-felt.binary                | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pink-jasper              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pink-jasper.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pink-plastic             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pink-plastic.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_polyethylene             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\polyethylene.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_polyurethane-foam        | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\polyurethane-foam.binary        | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pure-rubber              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pure-rubber.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_purple-paint             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\purple-paint.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_pvc                      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\pvc.binary                      | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-fabric               | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric.binary               | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-fabric2              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-fabric2.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-metallic-paint       | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-metallic-paint.binary       | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-phenolic             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-phenolic.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-plastic              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-plastic.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_red-specular-plastic     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\red-specular-plastic.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_silicon-nitrade          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\silicon-nitrade.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_silver-metallic-paint    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\silver-metallic-paint.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_silver-metallic-paint2   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\silver-metallic-paint2.binary   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_silver-paint             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\silver-paint.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_special-walnut-224       | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\special-walnut-224.binary       | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-black-phenolic  | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-black-phenolic.binary  | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-blue-phenolic   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-blue-phenolic.binary   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-green-phenolic  | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-green-phenolic.binary  | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-maroon-phenolic | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-maroon-phenolic.binary | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-orange-phenolic | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-orange-phenolic.binary | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-red-phenolic    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-red-phenolic.binary    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-violet-phenolic | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-violet-phenolic.binary | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-white-phenolic  | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-white-phenolic.binary  | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_specular-yellow-phenolic | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\specular-yellow-phenolic.binary | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_ss440                    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\ss440.binary                    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_steel                    | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\steel.binary                    | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_teflon                   | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\teflon.binary                   | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_tungsten-carbide         | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\tungsten-carbide.binary         | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_two-layer-gold           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\two-layer-gold.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_two-layer-silver         | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\two-layer-silver.binary         | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_violet-acrylic           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\violet-acrylic.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_violet-rubber            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\violet-rubber.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-acrylic            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-acrylic.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-diffuse-bball      | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-diffuse-bball.binary      | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-fabric             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-fabric.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-fabric2            | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-fabric2.binary            | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-marble             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-marble.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_white-paint              | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\white-paint.binary              | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_yellow-matte-plastic     | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\yellow-matte-plastic.binary     | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_yellow-paint             | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\yellow-paint.binary             | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_yellow-phenolic          | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\yellow-phenolic.binary          | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Material reflectance    | BRDF       | brdf_yellow-plastic           | C:\Users\mrkys\workspace\NTD-PL\data\BRDFDatabase\brdfs\yellow-plastic.binary           | 90x90x180x3  | 32x32x64x3      | (12,12,16,2) |
| Spectral natural scenes | Harvard HS | harvard_img3                  | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img3.mat                                  | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_img4                  | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img4.mat                                  | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_img5                  | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img5.mat                                  | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_img6                  | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\img6.mat                                  | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imga3                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imga3.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imga4                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imga4.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imga8                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imga8.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgc3                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgc3.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgc6                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgc6.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgd0                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgd0.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgd1                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgd1.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgd5                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgd5.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgd6                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgd6.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg0                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg0.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg1                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg1.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg2                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg2.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg3                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg3.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg4                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg4.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg5                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg5.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg6                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg6.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg7                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg7.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg8                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg8.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgg9                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgg9.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgh4                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh4.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgh5                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh5.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgh6                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh6.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
| Spectral natural scenes | Harvard HS | harvard_imgh7                 | C:\Users\mrkys\workspace\NTD-PL\data\CZ_hsdbi\imgh7.mat                                 | 1040x1392x31 | 128x128x31      | (32,32,8)    |
