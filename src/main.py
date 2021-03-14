import math
import numpy as np

from lib.runners import run_oned_convection, run_oned_linear_convection, run_bouss_fin_diff_poiss

def oned():
    # Number of x elements
    nx = 201
    # Number of time steps
    nt = nx
    x_min = -np.pi
    x_max = np.pi
    # Wave speed
    c = 1
    dx = 2 / (nx - 1)
    dt = 2 / (nx - 1)
    # Execute and plot 1D Linear Convection
    run_oned_linear_convection(
        nx=nx,
        nt=nx,
        x_min=x_min,
        x_max=x_max,
        c=c,
        dx=dx,
        dt=dt,
    )
    # Execute and plot 1D Convection
    run_oned_convection(
        nx=nx,
        nt=nx,
        x_min=x_min,
        x_max=x_max,
        dx=dx,
        dt=dt
    )


def main():
    # oned()
    # Bouss diff quella la
    nx = 101
    ny = 101
    final_time = 0.5
    epsilon = 0.5
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = epsilon * dx
    nt = math.floor(final_time / dt)
    # Richardson number
    b = 0.5
    pi = 2

    run_bouss_fin_diff_poiss(
        nx=nx,
        ny=ny,
        nt=nt,
        final_time=final_time,
        dt=dt,
        dx=dx,
        dy=dx,
        epsilon=epsilon,
        b=b,
        pi=pi
    )


if __name__ == "__main__":
    main()
