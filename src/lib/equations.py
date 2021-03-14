def oned_linear_convection(u, nx, c, dt, dx):
    """ 1D Linear convection method. """
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


def bouss_fin_diff_poiss(y, b, nx, dt, dx, ny, dy, om, th, psi):
    """
    I don't know really
    """
    omn = om.copy()
    thn = th.copy()
    psin = psi.copy()
    row, col = om.shape
    for j in range(1, row):
        om[j, 1:] = (
            omn[j, 1:]
            - b * (dt / dx) * (thn[j, 1:] - thn[j, :-1])
            - (dt / dx) * y[j] * (omn[j, 1:] - omn[j, :-1])
        )
        om[0, :] = 0
        om[-1, :] = 0
        om[:, 0] = (
            omn[:, 0]
            - b * (dt / dx) * (thn[:, 0] - thn[:, -2])
            - (y[j] * dt / dx * (omn[:, 0] - omn[:, -2]))
        )
        om[:, -1] = om[:, 0]

        th[j, 1:] = (
            thn[j, 1:]
            - (dt / dx) * (psin[j, 1:] - psin[j, :-1])
            - (dt / dx) * y[j] * (thn[j, 1:] - thn[j, :-1])
        )
        th[0, :] = 0
        th[-1, :] = 0
        th[:, 0] = (
            thn[:, 0]
            - (dt / dx) * (psin[:, 0] - psin[:, -2])
            - (y[j] * dt / dx * (thn[:, 0] - thn[:, -2]))
        )
        th[:, -1] = th[:, 0]

    app = 1000

    for it in range(app):

        psind = psin.copy()

        psin[1:-1, 1:-1] = (
            (psind[1:-1, 2:] + psind[1:-1, :-2]) * dy ** 2
            + (psind[2:, 1:-1] + psind[:-2, 1:-1]) * dx ** 2
            - om[1:-1, 1:-1] * dx ** 2 * dy ** 2
        ) / (2 * (dx ** 2 + dy ** 2))

        psin[0, :] = 0
        psin[ny - 1, :] = 0
        psin[:, 0] = psin[:, nx - 1]

    psi = psin
