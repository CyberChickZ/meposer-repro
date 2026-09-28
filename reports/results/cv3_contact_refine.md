Test-time wrist IK: shoulders/collars/elbows optimised so that the FK wrists match controller pose x one fixed offset estimated on the fold's training subjects. MPJPE cm per held-out subject, last epoch of each fold.

| model | wrist IK | 0000 | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | 0009 | MPJPE mean +- std (subjects common to all rows) | HandPE mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | - | 2.09 | 8.54 | 2.64 | 3.47 | 3.21 | 3.44 | 12.70 | 2.03 | 3.96 | 2.21 | 5.24 +- 4.01 | 13.04 |
| full | IK | 2.07 | 4.99 | 2.35 | 2.71 | 2.62 | 2.86 | 8.78 | 1.64 | 2.22 | 1.71 | 3.78 +- 2.48 | 2.36 |
| aug | - | 2.24 | 3.94 | 2.95 | - | - | 3.62 | 5.91 | 2.50 | - | - | 3.53 +- 1.22 | 6.29 |
| aug | IK | 2.15 | 3.63 | 2.48 | - | - | 2.88 | 4.48 | 1.83 | - | - | 2.91 +- 0.90 | 2.34 |
| aug_noise | - | 2.40 | 3.87 | 3.28 | - | - | 3.79 | 5.99 | 2.38 | - | - | 3.62 +- 1.22 | 6.43 |
| aug_noise | IK | 2.28 | 3.62 | 2.75 | - | - | 3.00 | 4.54 | 1.83 | - | - | 3.00 +- 0.89 | 2.34 |
| contact | - | 2.08 | 5.52 | 2.94 | - | - | 3.62 | 5.35 | 2.44 | - | - | 3.66 +- 1.34 | 7.80 |
| contact | IK | 2.05 | 4.14 | 2.52 | - | - | 3.01 | 3.97 | 1.65 | - | - | 2.89 +- 0.92 | 2.34 |
