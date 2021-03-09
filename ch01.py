import numpy

from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation


def sin_data(nx, x_min=-numpy.pi, x_max=numpy.pi):
    return numpy.sin(numpy.linspace(x_min, x_max, nx))


def gauss_data(nx, x_min=1, x_max=50, std=None):
    x = numpy.linspace(x_min, x_max, nx)
    mean = numpy.mean(x)
    if std is None:
        std = numpy.std(x)
    return numpy.pi * std * numpy.exp(-0.5 * ((x - mean) / std) ** 2)


def generate_initial_data(nx: int, dx: float) -> numpy.array:
    """ Generate the array used as initial conditions. """
    data = numpy.zeros(nx)
    # Set initial values, 2 for x >=0.5 and x <= 1.0
    # The index of `x == 0.5` is 0.5 / dx, and so for 1.0.
    # Add one to compensate for the truncation using `int`
    # in the first index
    data[int(0.5 / dx) : int(1 / dx + 1)] = 2
    return data


def run(method, nx, nt, x_min, x_max, dx, **kwargs):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = generate_initial_data(nx, dx)
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
    nx = 200
    # Number of time steps
    nt = 512
    # Wave speed
    c = 1
    # Delta x
    dx = 2 / (nx - 1)
    # Delta time.
    dt = dx / 2

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
    # run(oned_convection, nx, nt, -numpy.pi, numpy.pi, dt=dt, dx=dx)


if __name__ == "__main__":
    main()
