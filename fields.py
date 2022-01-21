import cupy as cp
import variables as var


class Static:
    """ Class for static fields governed by Gauss's law, here E_x """
    def __init__(self, resolution):
        self.electric_x = var.SpaceScalar(resolution=resolution)

    def gauss(self, distribution, grid, invert=True):
        # Compute zeroth moment, integrate(c_n(v)dv)
        distribution.compute_zero_moment(grid=grid)

        # Adjust for charge neutrality
        distribution.moment0.arr_spectral[grid.x.zero_idx] -= 1.0

        # Compute field spectrum
        self.electric_x.arr_spectral = (-1j * grid.charge_sign *
                                        cp.nan_to_num(cp.divide(distribution.moment0.arr_spectral,
                                                                grid.x.device_wavenumbers)))

        if invert:
            self.electric_x.inverse_fourier_transform()

    def compute_field_energy(self, grid):
        self.electric_x.inverse_fourier_transform()
        return self.electric_x.integrate_energy(grid=grid)


class Dynamic:
    """ Class for dynamic fields described by Ampere/Faraday laws, here E_y and B_z """
    def __init__(self, resolution):
        self.electric_y = var.SpaceScalar(resolution=resolution)
        self.magnetic_z = var.SpaceScalar(resolution=resolution)

    def initialize(self, grid, eigenvalue):
        # Set eigenmode
        self.eigenmode(grid=grid, amplitude=1.0e-3, wavenumber=0.1, eigenvalue=eigenvalue)  # ? 1.54j, 0.736j
        # Fourier-transform
        self.magnetic_z.fourier_transform()
        self.electric_y.fourier_transform()

    def eigenmode(self, grid, amplitude, wavenumber, eigenvalue):
        # Nodal values
        self.magnetic_z.arr_nodal = cp.real(amplitude * cp.exp(1j * wavenumber * grid.x.device_arr))
        self.electric_y.arr_nodal = cp.real(eigenvalue * amplitude * cp.exp(1j * wavenumber * grid.x.device_arr))

    def compute_magnetic_energy(self, grid):
        self.magnetic_z.inverse_fourier_transform()
        return self.magnetic_z.integrate_energy(grid=grid) / (0.3 ** 2.0)

    def compute_electric_energy(self, grid):
        self.electric_y.inverse_fourier_transform()
        return self.electric_y.integrate_energy(grid=grid)
