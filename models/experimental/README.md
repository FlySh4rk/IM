# Experiments

## Simple segmentation problem

Separate tooth from crown.

| Num | descr                                          | simple   | rotated          |
|-----|------------------------------------------------|----------|------------------|
| 1   | norm, no avg,no avg delta, cross               | A10=70%  | -                |
| 2   | norm, avg,no avg_delta, cross                  | A10=100% | A10=75%          |
| 3   | no norm, avg, no avg delta, cross              | A10=100% | A10=50%          |
| 4   | norm, avg, non avg delta, variadic, 16 probes  | ?        | A10=72%          |
| 5   | 16 variadic probes + dense summary             |          | A10=76%          |
| 6   | 4 variadic probes with polar coords for probes |          | A10=70%          |
| 7   | CrossConvV2                                    |          | A10=75%          |
| 8   | CrossConvV2 X 2                                |          | A10=92%, A20=94% |
| 9   | CrossConvV2 X 3                                |          | A10=92%, A20=92% |
| 10  | Fat CrossConvV3                                |          | A10=99% (!!!!)   |

