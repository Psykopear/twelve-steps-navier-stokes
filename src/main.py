import numpy

from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation

from lib.data import square, sin, gauss
from lib.equations import oned_convection, oned_linear_convection


def run(method, nx, nt, x_min, x_max, dx, **kwargs):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = square(nx, dx)
    # u = sin_data(nx)
    # u = gauss_data(nx, std=0.5)
    # Apply `method` for `nt` times
    # and save each frame to create an animation
    for i in range(nt):
        method(u, nx, dx=dx, **kwargs)
        frames.append(ax.plot(numpy.linspace(x_min, x_max, nx), u, "r"))
    # Now create the animation at 1 frame every 25 milliseconds
    animation = ArtistAnimation(fig, frames, repeat=False, interval=25)
    # And render to an mp4
    animation.save(f"{method.__name__}.mp4")


def main():
    # Number of x elements
    nx = 201
    # Number of time steps
    nt = 201
    # Wave speed, used in linear convection
    c = 1
    # Delta x
    dx = 2 / (nx - 1)
    # Delta time.
    dt = dx

    # Execute and plot 1D Linear Convection
    run(
        method=oned_linear_convection,
        nx=nx,
        nt=nt,
        x_min=-numpy.pi,
        x_max=numpy.pi,
        c=c,
        dt=dt,
        dx=dx,
    )
    # Execute and plot 1D Convection
    run(
        method=oned_convection,
        nx=nx,
        nt=nt,
        x_min=-numpy.pi,
        x_max=numpy.pi,
        dt=dt,
        dx=dx,
    )


if __name__ == "__main__":
    main()
