from pathlib import Path

import time
from predict_modulation import predict_modulation
import numpy as np


start = time.monotonic()
predict_modulation(
    Path("src/xsw-scripts/fpfpp/"),
    Path("src/xsw-scripts/fpfpp/"),
    1,
    1,
    18,
    True,
    8,
    1,
    0,
    0.5,
    0,
    np.array([1, 1, 1]),
    np.array([3.6149, 3.6149, 3.6149, 90, 90, 90]),
    np.array(
        [
            [29, 0.0, 0.0, 0.0, 1],
            [29, 0.5, 0.5, 0.0, 1],
            [29, 0.5, 0.0, 0.5, 1],
            [29, 0.0, 0.5, 0.5, 1],
        ]
    ),
    0.2,
)
stop = time.monotonic()
print(f"Time: {stop - start}")
