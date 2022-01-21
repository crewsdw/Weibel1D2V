import cupy as cp


class SpaceScalar:
    """ examples: fields, density, etc. """

    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * self.arr_nodal ** 2.0
        arr_add = cp.append(arr, arr[0])
        return trapz(arr_add, grid.x.dx)


class Distribution:
    def __init__(self, resolutions, order):
        self.resolutions, self.order = resolutions, order

        self.arr_nodal, self.arr_spectral = None, None
        self.moment0, self.moment2 = SpaceScalar(resolution=resolutions[0]), SpaceScalar(resolution=resolutions[0])
        self.moment_v1 = SpaceScalar(resolution=resolutions[0])

    def fourier_transform(self):
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, axis=0, norm='forward')

    def compute_zero_moment(self, grid):
        self.moment0.arr_spectral = grid.zero_moment(variable=self.arr_spectral)
        self.moment0.inverse_fourier_transform()

    def compute_moment_v1(self, grid):
        self.moment_v1.arr_spectral = grid.zero_moment(variable=(grid.v.device_arr[None, None, None, :, :] *
                                                                 self.arr_spectral))
        self.moment_v1.inverse_fourier_transform()

    def compute_second_moment(self, grid):
        self.moment2.arr_spectral = grid.second_moment(variable=self.arr_spectral)
        self.moment2.inverse_fourier_transform()

    def total_density(self, grid):
        self.compute_zero_moment(grid=grid)
        return self.moment0.integrate(grid=grid)

    def total_thermal_energy(self, grid):
        self.compute_second_moment(grid=grid)
        return 0.5 * self.moment2.integrate(grid=grid)

    def nodal_flatten(self):
        return self.arr_nodal.reshape(self.resolutions[0],
                                      self.resolutions[1] * self.order, self.resolutions[2] * self.order)

    def spectral_flatten(self):
        return self.arr_spectral.reshape(self.arr_spectral.shape[0],
                                         self.resolutions[1] * self.order, self.resolutions[2] * self.order)

    def initialize(self, grid, eigenvalue):
        """ Initialize a distribution of transverse beams """
        # grid-likes
        ix, iu, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        # maxwellians
        max_u = cp.exp(-grid.u.device_arr ** 2.0 / 2) / cp.sqrt(2 * cp.pi)
        max_v = 0.5 * (cp.exp(-(grid.v.device_arr - 1) ** 2.0 / 2) +
                       cp.exp(-(grid.v.device_arr + 1) ** 2.0 / 2)) / cp.sqrt(2 * cp.pi)
        # equilibrium distribution
        self.arr_nodal = cp.tensordot(ix, cp.tensordot(max_u, max_v, axes=0), axes=0)
        # perturbation# 1.54j) # 0.736j
        self.arr_nodal += self.kinetic_eigenmode(grid=grid, amplitude=1.0e-3, wavenumber=0.1, eigenvalue=eigenvalue)
        self.fourier_transform()

    def kinetic_eigenmode(self, grid, amplitude, wavenumber, eigenvalue):
        """ The eigenvalue is a complex phase velocity """
        # maxwellians
        max_u = cp.exp(-grid.u.device_arr ** 2.0 / 2) / cp.sqrt(2.0 * cp.pi)
        max_v = 0.5 * (cp.exp(-(grid.v.device_arr - 1) ** 2.0 / 2) +
                       cp.exp(-(grid.v.device_arr + 1) ** 2.0 / 2)) / cp.sqrt(2 * cp.pi)
        # gradients
        dfu_du = -grid.u.device_arr * cp.exp(-grid.u.device_arr ** 2.0 / 2) / cp.sqrt(2 * cp.pi)
        dfv_dv = 0.5 * (-(grid.v.device_arr - 1) * cp.exp(-(grid.v.device_arr - 1) ** 2.0 / 2) -
                        (grid.v.device_arr + 1) * cp.exp(-(grid.v.device_arr + 1) ** 2.0 / 2)) / cp.sqrt(2 * cp.pi)
        df_du = cp.tensordot(dfu_du, max_v, axes=0) + 0j
        df_dv = cp.tensordot(max_u, dfv_dv, axes=0) + 0j
        # perturbation
        mode = 1j * (amplitude / wavenumber) * (
                (grid.v.device_arr[None, None, :, :] / (eigenvalue - grid.u.device_arr[:, :, None, None])) * df_du +
                df_dv
        )
        return cp.real(cp.tensordot(cp.exp(1j * wavenumber * grid.x.device_arr), mode, axes=0))


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
