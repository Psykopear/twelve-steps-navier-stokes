import numpy

from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation


def generate_initial_data(nx: int, dx: float) -> numpy.array:
    """
    Generate the array used as initial conditions.

    Filled with `nx` zeros
    """
    data = numpy.zeros(nx)
    # Set initial values, 2 for x >=0.5 and x <= 1.0
    # The index of `x == 0.5` is 0.5 / dx, and so for 1.0.
    # Add one to compensate for the truncation using `int`
    # in the first index
    data[int(0.5 / dx) : int(1 / dx + 1)] = 2
    return data


def oned_linear_convection(u, nx, c, dt, dx):
    """
    1D Linear convection method.

    The array is modified in place.
    """
    un = u.copy()
    for i in range(2, nx - 1):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])


def oned_convection(u, nx, dt, dx):
    """
    1D convection method.

    Like the previous one, but removing linearity:
    we will use u[i] instead of c
    """
    un = u.copy()
    for i in range(2, nx - 1):
        u[i] = un[i] - u[i] * dt / dx * (un[i] - un[i - 1])


def run_oned_nonlinear(nx, dx, nt, dt):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = generate_initial_data(nx, dx)
    # Apply 1D linear convection `nt` times
    # and save each frame to create an animation
    for i in range(nt):
        oned_convection(u, nx, dt, dx)
        frames.append(ax.plot(numpy.linspace(0, 2, nx), u, 'r'))
    # Now create the animation at 1 frame every `interval` milliseconds
    animation = ArtistAnimation(fig, frames, repeat=False, interval=10)
    # And render to an mp4
    animation.save("oned_convection.mp4")


def run_oned_linear(nx, dx, nt, dt, c):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = generate_initial_data(nx, dx)
    # Apply 1D linear convection `nt` times
    # and save each frame to create an animation
    for i in range(nt):
        oned_linear_convection(u, nx, c, dt, dx)
        frames.append(ax.plot(numpy.linspace(0, 2, nx), u, 'r'))
    # Now create the animation at 1 frame every 25 milliseconds
    animation = ArtistAnimation(fig, frames, repeat=False, interval=25)
    # And render to an mp4
    animation.save("oned_linear_convection.mp4")


def main():
    # Number of x elements
    nx = 512
    # Number of time steps
    nt = 512
    # Wave speed
    c = 1
    # Delta x
    dx = 2 / nx
    # dx = 0.025
    # Delta time.
    dt = dx / 2
    # dt = 0.01

    # Execute and plot 1D Linear Convection
    run_oned_linear(nx, dx, nt, dt, c)
    # Execute and plot 1D Convection
    # run_oned_nonlinear(nx, dx, nt, dt)


if __name__ == "__main__":
    main()
