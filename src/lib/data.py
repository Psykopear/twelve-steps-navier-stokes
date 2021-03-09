import numpy as np
import numpy.typing as npt

from typing import Optional, TypeVar, Any


def sin(nx: int, x_min: float = -np.pi, x_max: float = np.pi) -> np.ndarray:
    return np.sin(np.linspace(x_min, x_max, nx))


def gauss(
    nx: int, x_min: float = 1.0, x_max: float = 50.0, std: Optional[np.number] = None
) -> np.ndarray:
    x = np.linspace(x_min, x_max, nx)
    mean = np.mean(x)
    if std is None:
        std = np.std(x)
    return np.pi * std * np.exp(-0.5 * ((x - mean) / std) ** 2)


def square(nx: int, dx: float) -> np.ndarray:
    """ Generate the array used as initial conditions. """
    data = np.zeros(nx)
    # Set initial values, 2 for x >=0.5 and x <= 1.0
    # The index of `x == 0.5` is 0.5 / dx, and so for 1.0.
    # Add one to compensate for the truncation using `int`
    # in the first index
    data[int(0.5 / dx) : int(1 / dx + 1)] = 2
    return data
