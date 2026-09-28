# SMPL body model

This repo does not ship SMPL. Download `SMPL_NEUTRAL.pkl` yourself from
<https://smpl.is.tue.mpg.de/> (free academic registration, their licence
forbids redistribution) and convert it once:

```bash
python tools/convert_smpl_pkl.py /path/to/SMPL_NEUTRAL.pkl assets/smpl/SMPL_NEUTRAL.npz
```

The converter unpickles the official file without needing `chumpy` (its objects are replaced by plain arrays) and
densifies the sparse joint regressor; it works on Python >= 3.11 and is covered by a synthetic round-trip check.
