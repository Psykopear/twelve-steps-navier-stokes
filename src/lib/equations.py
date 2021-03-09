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
