# Non-HSI D_link Diagnostic

- Mode: `full`
- Output directory: `results\nonhsi_dlink_diagnostic\20260507_000000_full68`
- Elapsed wall time: 76.6 s
- Seed: 0
- Tucker iterations: 180
- NTD-PL iterations: 180
- NTD-PL polynomial degree: 4
- NTD-PL ridge lambda_beta: 1e-06
- Link refresh sample size: 400000 (<=0 means all entries)

## Data Paths And Preprocessing

- CBSD68 root=C:\Users\mrkys\workspace\NTD-PL\data\CBSD68\train; units=68; resize=96x96 RGB.
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
| Natural images | CBSD68    | 68        | 0.0911        | 0.1355      | 0.8471              | 8.9750    | 7.4190      | 4.1735           | 8.3972           | 14.5987           | 23            | 23           |
| Natural images | CIFAR-10  | 1000      | 0.0080        | 0.0160      | 0.9336              | 3.4732    | 2.3521      | -0.0074          | 2.5854           | 7.8519            | 135           | 548          |
| Object-view    | COIL-100  | 100       | 0.2079        | 0.2415      | 0.9322              | 5.7617    | 5.3982      | 2.7708           | 5.4640           | 9.1409            | 34            | 33           |
| Object-view    | smallNORB | 25        | 0.2637        | 0.3070      | 0.9454              | 10.1941   | 9.8360      | 6.9208           | 9.9072           | 14.1634           | 9             | 9            |

## Representative Units

| domain         | dataset   | unit_id            | unit_label             | selection             | diagnostic_label | d_link_db | tucker_rmse | ntdpl_rmse | rmse_gain_pct | rank         | shape      |
| -------------- | --------- | ------------------ | ---------------------- | --------------------- | ---------------- | --------- | ----------- | ---------- | ------------- | ------------ | ---------- |
| Natural images | CBSD68    | cbsd68_062         | 0061                   | bottom-2 by D_link    | boundary         | 0.0016    | 0.0250      | 0.0250     | 0.1387        | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_054         | 0053                   | bottom-2 by D_link    | boundary         | 0.0019    | 0.0176      | 0.0173     | 1.1928        | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_057         | 0056                   | top-2 by D_link       | effective        | 0.4873    | 0.0614      | 0.0480     | 21.7854       | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_064         | 0063                   | top-2 by D_link       | effective        | 1.1151    | 0.0447      | 0.0271     | 39.4127       | (32,32,3)    | 96x96x3    |
| Natural images | CBSD68    | cbsd68_067         | 0066                   | median-near by D_link | moderate         | 0.0928    | 0.0432      | 0.0381     | 11.8877       | (32,32,3)    | 96x96x3    |
| Natural images | CIFAR-10  | cifar10_test_06970 | class0_test6970        | bottom-2 by D_link    | boundary         | -0.1362   | 0.0008      | 0.0008     | -4.4553       | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_09848 | class2_test9848        | bottom-2 by D_link    | boundary         | -0.0775   | 0.0011      | 0.0011     | -0.6788       | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_05446 | class6_test5446        | top-2 by D_link       | effective        | 0.4406    | 0.0085      | 0.0047     | 45.3123       | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_04753 | class2_test4753        | top-2 by D_link       | effective        | 0.7890    | 0.0140      | 0.0070     | 49.9183       | (20,20,3)    | 32x32x3    |
| Natural images | CIFAR-10  | cifar10_test_08046 | class8_test8046        | median-near by D_link | boundary         | 0.0080    | 0.0047      | 0.0045     | 3.6683        | (20,20,3)    | 32x32x3    |
| Object-view    | COIL-100  | coil_obj088        | object 88              | bottom-2 by D_link    | boundary         | 0.0172    | 0.0280      | 0.0276     | 1.5124        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj082        | object 82              | bottom-2 by D_link    | boundary         | 0.0241    | 0.0392      | 0.0389     | 0.9231        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj044        | object 44              | top-2 by D_link       | effective        | 0.7157    | 0.0958      | 0.0837     | 12.6724       | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj091        | object 91              | top-2 by D_link       | effective        | 0.8746    | 0.0751      | 0.0650     | 13.3371       | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | COIL-100  | coil_obj013        | object 13              | median-near by D_link | moderate         | 0.2065    | 0.0500      | 0.0473     | 5.3928        | (8,12,12,2)  | 36x48x48x3 |
| Object-view    | smallNORB | smallnorb_c3_i4    | category 3, instance 4 | bottom-2 by D_link    | boundary         | 0.0772    | 0.0315      | 0.0301     | 4.5103        | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c3_i7    | category 3, instance 7 | bottom-2 by D_link    | boundary         | 0.0857    | 0.0275      | 0.0259     | 6.1372        | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c2_i9    | category 2, instance 9 | top-2 by D_link       | effective        | 0.6914    | 0.0471      | 0.0400     | 15.0928       | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c2_i7    | category 2, instance 7 | top-2 by D_link       | effective        | 1.0485    | 0.0478      | 0.0380     | 20.5253       | (12,24,24,2) | 18x64x64x2 |
| Object-view    | smallNORB | smallnorb_c2_i4    | category 2, instance 4 | median-near by D_link | moderate         | 0.2637    | 0.0452      | 0.0407     | 9.8360        | (12,24,24,2) | 18x64x64x2 |

## Failures

No failed units.

## Main Conclusions

- Natural images mostly weak/modest shared-link yield: yes.
- Object-view tensors show clearer effective units: yes.
- Table 3 update recommendation: use this output as candidate evidence only; main.tex was not modified.

## Reproduction Command

```powershell
python scripts/run_nonhsi_dlink_diagnostic.py --mode full --datasets cbsd68,cifar10,coil100,smallnorb --jobs 6 --seed 0 --n-iter-max 180 --tucker-n-iter-max 180 --p-max 4 --outdir results\nonhsi_dlink_diagnostic\20260507_000000_full68
```
