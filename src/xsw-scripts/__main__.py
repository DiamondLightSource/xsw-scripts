from pathlib import Path

from predict_modulation import predict_modulation
from q_param import q_param
import numpy as np

predict_modulation(
    Path("src/xsw-scripts/fpfpp/"),
    1,
    1,
    18,
    1,
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
