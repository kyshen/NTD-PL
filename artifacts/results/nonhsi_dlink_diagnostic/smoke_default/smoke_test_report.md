# Non-HSI D_link Diagnostic

- Mode: `smoke`
- Output directory: `results\nonhsi_dlink_diagnostic\smoke_default`
- Elapsed wall time: 7.4 s
- Seed: 0
- Tucker iterations: 180
- NTD-PL iterations: 180
- NTD-PL polynomial degree: 4
- NTD-PL ridge lambda_beta: 1e-06
- Link refresh sample size: 400000 (<=0 means all entries)

## Data Paths And Preprocessing

- CBSD68 root=C:\Users\mrkys\workspace\NTD-PL\data\CBSD68\validation; units=16; resize=96x96 RGB.
- CIFAR-10 source=C:\Users\mrkys\workspace\NTD-PL\data\cifar-10-python.tar.gz; class-balanced units=1000; shape=32x32x3.
- COIL-100 root=C:\Users\mrkys\workspace\NTD-PL\data\coil-100\coil-100; objects=100; views=36 via angles 0..350 step 10; resize=48x48 RGB.
- smallNORB root=C:\Users\mrkys\workspace\NTD-PL\data\smallNORB; object instances=25; elevation=4 lighting=0 azimuth=18; resize=64x64 stereo.
- All units are max-normalized independently before fitting.
- Natural-image units are single images; object-view units are per-object/per-instance multi-view tensors.
- No cross-class or cross-object giant tensor is used for the main diagnostic unit.

## Unit Definitions And Ranks

- CBSD68: one 96x96x3 image, rank (32,32,3).
- CIFAR-10: one 32x32x3 test image, rank (20,20,3).
- COIL-100: one 36x48x48x3 object-view tensor, rank (8,12,12,2).
- smallNORB: one 18x64x64x2 instance-view tensor, rank (12,24,24,2).

## Label Rule

Diagnostic labels are assigned only from D_link, not from NTD-PL gain: boundary is the bottom D_link tercile or D_link <= 0.01 dB; effective is the top tercile with D_link > 0.03 dB; all remaining successful units are moderate.

## Dataset Summary

| domain         | dataset   | num_units | median_d_link | mean_d_link | spearman_dlink_gain | mean_gain | median_gain | low_tercile_gain | mid_tercile_gain | high_tercile_gain | num_effective | num_boundary |
| -------------- | --------- | --------- | ------------- | ----------- | ------------------- | --------- | ----------- | ---------------- | ---------------- | ----------------- | ------------- | ------------ |
| Natural images | CBSD68    | 3         | 0.0735        | 0.1554      | 0.5000              | 9.6273    | 7.5028      | 7.5028           | 4.6911           | 16.6879           | 1             | 1            |
| Natural images | CIFAR-10  | 3         | 0.0314        | 0.0339      | 0.5000              | 5.7789    | 5.5036      | 3.7634           | 8.0696           | 5.5036            | 1             | 1            |
| Object-view    | COIL-100  | 3         | 0.3114        | 0.4373      | 1.0000              | 8.3370    | 6.4838      | 5.1902           | 6.4838           | 13.3371           | 1             | 1            |
| Object-view    | smallNORB | 3         | 0.1569        | 0.2098      | 1.0000              | 7.6002    | 6.2365      | 4.5103           | 6.2365           | 12.0539           | 1             | 1            |

## Representative Units

| domain         | dataset   | unit_id            | unit_label             | selection             | diagnostic_label | d_link_db | tucker_rmse | ntdpl_rmse | rmse_gain_pct | rank         | shape      |
| -------------- | --------- | ------------------ | ---------------------- | --------------------- | ---------------- | --------- | ----------- | ---------- | ------------- | ------------ | ---------- |
| Natural images | CBSD68    | cbsd68_010         | 41029                  | bottom-2 by D_link    | boundary         | 0.0520    | 0.0350      | 0.0324     | 7.5028        | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_009         | 388006                 | bottom-2 by D_link    | moderate         | 0.0735    | 0.0523      | 0.0498     | 4.6911        | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_009         | 388006                 | top-2 by D_link       | moderate         | 0.0735    | 0.0523      | 0.0498     | 4.6911        | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_012         | 48017                  | top-2 by D_link       | effective        | 0.3408    | 0.0490      | 0.0408     | 16.6879       | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_009         | 388006                 | median-near by D_link | moderate         | 0.0735    | 0.0523      | 0.0498     | 4.6911        | (32,32,3)    | 96x96x3    |
| Natural images | CIFAR-10  | cifar10_test_00829 | class3_test829         | bottom-2 by D_link    | boundary         | 0.0106    | 0.0056      | 0.0054     | 3.7634        | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_00148 | class5_test148         | bottom-2 by D_link    | moderate         | 0.0314    | 0.0057      | 0.0052     | 8.0696        | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_00148 | class5_test148         | top-2 by D_link       | moderate         | 0.0314    | 0.0057      | 0.0052     | 8.0696        | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_00392 | class6_test392         | top-2 by D_link       | effective        | 0.0597    | 0.0176      | 0.0167     | 5.5036        | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_00148 | class5_test148         | median-near by D_link | moderate         | 0.0314    | 0.0057      | 0.0052     | 8.0696        | (20,20,3)    | 32x32x3    |
| Object-view    | COIL-100  | coil_obj064        | object 64              | bottom-2 by D_link    | boundary         | 0.1259    | 0.0378      | 0.0359     | 5.1902        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj051        | object 51              | bottom-2 by D_link    | moderate         | 0.3114    | 0.0623      | 0.0583     | 6.4838        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj051        | object 51              | top-2 by D_link       | moderate         | 0.3114    | 0.0623      | 0.0583     | 6.4838        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj091        | object 91              | top-2 by D_link       | effective        | 0.8746    | 0.0751      | 0.0650     | 13.3371       | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj051        | object 51              | median-near by D_link | moderate         | 0.3114    | 0.0623      | 0.0583     | 6.4838        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | smallNORB | smallnorb_c3_i4    | category 3, instance 4 | bottom-2 by D_link    | boundary         | 0.0772    | 0.0315      | 0.0301     | 4.5103        | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c3_i6    | category 3, instance 6 | bottom-2 by D_link    | moderate         | 0.1569    | 0.0284      | 0.0266     | 6.2365        | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c3_i6    | category 3, instance 6 | top-2 by D_link       | moderate         | 0.1569    | 0.0284      | 0.0266     | 6.2365        | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c2_i8    | category 2, instance 8 | top-2 by D_link       | effective        | 0.3954    | 0.0387      | 0.0340     | 12.0539       | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c3_i6    | category 3, instance 6 | median-near by D_link | moderate         | 0.1569    | 0.0284      | 0.0266     | 6.2365        | (12,24,24,2) | 18x64x64x2 |

## Failures

No failed units.

## Main Conclusions

- Natural images mostly weak/modest shared-link yield: yes.
- Object-view tensors show clearer effective units: yes.
- Table 3 update recommendation: use this output as candidate evidence only; main.tex was not modified.

## Reproduction Command

```powershell
python scripts/run_nonhsi_dlink_diagnostic.py --mode smoke --datasets cbsd68,cifar10,coil100,smallnorb --jobs 4 --seed 0 --n-iter-max 180 --tucker-n-iter-max 180 --p-max 4 --outdir results\nonhsi_dlink_diagnostic\smoke_default
```
