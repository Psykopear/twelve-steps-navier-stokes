import numpy as np
import pylab as py

from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation

from .data import get_oned_data, InitialDataType
from .equations import oned_convection, oned_linear_convection, bouss_fin_diff_poiss


def run_oned_convection(method, nx, nt, x_min, x_max, dx, dt):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = get_oned_data(InitialDataType.SQUARE, nx=nx, dx=dx)
    # Apply `method` for `nt` times
    # and save each frame to create an animation
    for i in range(nt):
        oned_convection(u=u, nx=nx, dx=dx, dt=dt)
        frames.append(ax.plot(np.linspace(x_min, x_max, nx), u, "r"))
    # Now create the animation at 1 frame every 25 milliseconds
    animation = ArtistAnimation(fig, frames, repeat=False, interval=25)
    # And render to an mp4
    animation.save(f"{method.__name__}.mp4")


def run_oned_linear_convection(method, nx, nt, x_min, x_max, dx, dt, c):
    # Generate figure for animation
    fig = pyplot.figure()
    ax = fig.add_subplot()

    frames = []
    # Generate initial data
    u = get_oned_data(InitialDataType.SQUARE, nx=nx, dx=dx)
    # Apply `method` for `nt` times
    # and save each frame to create an animation
    for i in range(nt):
        oned_linear_convection(u=u, nx=nx, dx=dx, dt=dt, c=c)
        frames.append(ax.plot(np.linspace(x_min, x_max, nx), u, "r"))
    # Now create the animation at 1 frame every 25 milliseconds
    animation = ArtistAnimation(fig, frames, repeat=False, interval=25)
    # And render to an mp4
    animation.save(f"{method.__name__}.mp4")


def run_bouss_fin_diff_poiss(nx, ny, final_time, dx, dy, epsilon, dt, nt, b, pi):
    x = np.linspace(-pi / 2, pi / 2, nx)
    y = np.linspace(0, pi, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize condition for the vorticity at t=0
    om = np.zeros((ny, nx))

    # Initialize conditions for the density at t=0: a bump function
    th = np.zeros((ny, nx))

    # Parameters for a bump of an ellipsoid
    # semiax in x
    ex = 0.25
    # semiax in y
    ey = 1
    # radius
    r = 1
    # center in x
    cx = 0
    # center in x
    cy = pi / 2
    # scaling constant row, col = th.shape
    C = np.exp(1 / r)
    row, col = th.shape

    for j in range(1, row):
        for i in range(1, col):
            if (x[i] - cx) ** 2 / ex + (y[j] - cy) ** 2 / ey < r:
                th[j, i] = C * np.exp(
                    -1 / (r - ((x[i] - cx) ** 2 / ex + (y[j] - cy) ** 2 / ey))
                )
            else:
                th[j, i] = 0

    # Compute initial value of streamfunction
    psi = np.zeros((ny, nx))
    psiin = psi
    ## parameters
    maxom = 8  # maximumm value of vorticity for plots
    t = 0
    thmin = np.amin(th)
    thmax = np.amax(th)

    maxth = np.amax([abs(thmin), abs(thmax)])
    normth = round(np.linalg.norm(th * dx), 3)
    normom = round(np.linalg.norm(om * dx), 3)

    fig, ((ax1), (ax2)) = pyplot.subplots(1, 2, figsize=(11.5, 5))
    z1_plot = ax1.pcolor(X, Y, om, cmap="BrBG", vmin=-maxom, vmax=maxom, shading="auto")
    py.colorbar(z1_plot, ax=ax1)
    # py.xlabel('x'); py.ylabel('y');
    ax1.set_title(
        r"Vorticity at time $t=$"
        "" + str(t) + ". "
        r" $\left\|| \omega(t)\right\||_{2}=$"
        "" + str(normom) + ""
    )

    z2_plot = ax2.pcolor(X, Y, th, cmap="bwr", vmin=-maxth, vmax=maxth, shading="auto")
    py.colorbar(z2_plot, ax=ax2)
    # py.xlabel('x'); py.ylabel('y'); a
    ax2.set_title(
        r"Density at time $t=$"
        "" + str(t) + ". "
        r" $\left\|| \theta(t) \right\||_{2}=$"
        "" + str(normth) + ""
    )

    fig.tight_layout()
    pyplot.savefig("plotinitial.png")

    ## Compute maximum values to fix a scale
    psimin = np.amin(psiin)
    psimax = np.amax(psiin)

    maxpsi = np.amax([abs(psimin), abs(psimax)])
    normpsiin = round(np.linalg.norm(psiin * dx), 3)

    ## Plot of initial stream function
    pyplot.pcolor(X, Y, psiin, cmap="BrBG", vmin=-maxpsi, vmax=maxpsi, shading="auto")
    py.colorbar()
    py.xlabel("x")
    py.ylabel("y")
    py.title(
        "Initial streamfunction. "
        r" $\left\|| \psi^{in} \right\||_{2}=$"
        "" + str(normpsiin) + ""
    )

    # Initialize vectors to solve the system
    # Create a 1xn vector of 0's
    omn = np.zeros((ny, nx))
    thn = np.zeros((ny, nx))
    psin = np.zeros((ny, nx))
    # Final time divided by T
    TT = 10
    frames = []
    fig, ((ax1), (ax2)) = pyplot.subplots(1, 2, figsize=(11.5, 5))
    for k in range(TT):
        tn = t

        # Loop across number of time steps
        for n in range(nt + 1):
            bouss_fin_diff_poiss(
                y=y, b=b, nx=nx, dt=dt, dx=dx, ny=ny, dy=dy, om=om, th=th, psi=psi
            )

            t = tn + dt

            # Maximumm value of vorticity for plots
            maxom = 8
            thmin = np.amin(th)
            thmax = np.amax(th)

            maxth = np.amax([abs(thmin), abs(thmax)])
            normth = round(np.linalg.norm(th * dx), 3)
            normom = round(np.linalg.norm(om * dx), 3)

            z1_plot = ax1.pcolor(
                X, Y, om, cmap="BrBG", vmin=-maxom, vmax=maxom, shading="auto"
            )
            z2_plot = ax2.pcolor(
                X, Y, th, cmap="bwr", vmin=-maxth, vmax=maxth, shading="auto"
            )
            # ax1.set_title(
            #     f"Vorticity at time\t$t=${t}\t\t$\\left\\||\\omega(t)\\right\\||_{2}=${normom}",
            #     loc="left"
            # )
            # ax2.set_title(
            #     # f"Density at time\t$t=${t}\t\t$\\left\\||\\tetha(t)\\right\\||_{2}=${normth}",
            #     # loc="left"
            #     r"Density at time       "
            #     r"$t=$"
            #     "" + str(t) + "       "
            #     r" $\left\|| \theta(t) \right\||_{2}=$"
            #     "" + str(normth) + ""
            # )
            # py.colorbar(z2_plot, ax=ax2)
            # py.colorbar(z1_plot, ax=ax1)
            # pyplot.savefig(f"plot{k}.png")
            frames.append((z1_plot, z2_plot))
            # break
    ax1.set_title("Vorticity")
    ax2.set_title("Density")
    fig.tight_layout()
    animation = ArtistAnimation(fig, frames, repeat=False, interval=25)
    animation.save("prova.mp4")
