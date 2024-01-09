# xsw-scripts
Generating Xray Standing Wave Data.

## Running the example

Install dependencies in a venv

```bash
    python -m venv venv
    source venv/bin/activate
    pip install .
```

Run timed example
```bash
    python src/xsw-scripts/__main__.py
```
This will run with the arguments in \__main__.py .

## Usage

```python
import numpy as np
from xsw_scripts import predict_modulation
from pathlib import Path

fit_out = predict_modulation(
    data_dir = Path("src/xsw_scripts/fpfpp/"),
    coherent_fraction= 1,
    coherent_position=1 ,
    theta= 18,
    plotter = False,
    atom_type=8,
    principal_qn=1,
    azimuthal_qn=0,
    spin_qn=0.5,
    alphaB=0,
    hkl_index=(1, 1, 1),
    lps0=np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
    xyzs=np.array(
        [
            [29, 0.0, 0.0, 0.0, 1],
            [29, 0.5, 0.5, 0.0, 1],
            [29, 0.5, 0.0, 0.5, 1],
            [29, 0.0, 0.5, 0.5, 1],
        ]
    ),
    width=0.2,
    )
```