import numpy as np
import cupy as cp
import variables as var
# import plotter as my_plt
import time as timer


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class PhaseSpaceFlux:
    def __init__(self, resolutions, x_modes, order, charge_sign):
        resolutions[0] = x_modes
        self.resolutions = resolutions
        self.order = order
        self.charge_sign = charge_sign

        self.permutations = [(0, 1, 4, 2, 3),
                             (0, 1, 2, 3, 4)]

        # dimension-dependent lists of slices into the phase space
        self.boundary_slices = [[(slice(self.resolutions[0]), slice(self.resolutions[1]), 0,
                                  slice(self.resolutions[2]), slice(self.order)),
                                 (slice(self.resolutions[0]), slice(self.resolutions[1]), -1,
                                  slice(self.resolutions[2]), slice(self.order))],
                                [(slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), 0),
                                 (slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                                  slice(self.resolutions[2]), -1)]]
        self.boundary_slices_pad = [[(slice(self.resolutions[0]), slice(self.resolutions[1] + 2), 0,
                                      slice(self.resolutions[2]), slice(self.order)),
                                     (slice(self.resolutions[0]), slice(self.resolutions[1] + 2), -1,
                                      slice(self.resolutions[2]), slice(self.order))],
                                    [(slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2] + 2), 0),
                                     (slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                                      slice(self.resolutions[2] + 2), -1)]]
        self.flux_input_slices = [(slice(self.resolutions[0]), slice(1, self.resolutions[1] + 1), slice(self.order),
                                   slice(self.resolutions[2]), slice(self.order)),
                                  (slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                                   slice(1, self.resolutions[2] + 1), slice(self.order))]
        self.pad_slices = [(slice(self.resolutions[0]), slice(1, self.resolutions[1] + 1),
                            slice(self.resolutions[2]), slice(self.order)),
                           (slice(self.resolutions[0]), slice(self.resolutions[1]), slice(self.order),
                            slice(1, self.resolutions[2] + 1))]

        # Array sizes for allocation
        self.num_flux_sizes = [(self.resolutions[0], self.resolutions[1], 2, self.resolutions[2], self.order),
                               (self.resolutions[0], self.resolutions[1], self.order, self.resolutions[2], 2)]
        self.padded_flux_sizes = [(self.resolutions[0],
                                   self.resolutions[1] + 2, self.order, self.resolutions[2], self.order),
                                  (self.resolutions[0],
                                   self.resolutions[1], self.order, self.resolutions[2] + 2, self.order)]
        self.sub_elements = [2, 4]
        self.directions = [1, 3]

        # arrays
        self.flux_ex = var.Distribution(resolutions=resolutions, order=order)
        self.flux_ey = var.Distribution(resolutions=resolutions, order=order)
        self.flux_bz = var.Distribution(resolutions=resolutions, order=order)
        # total output
        self.output = var.Distribution(resolutions=resolutions, order=order)
        # Initialize zero-pads
        self.pad_field = None
        self.pad_spectrum = None

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros(grid.x.modes + grid.x.pad_width) + 0j
        self.pad_spectrum = cp.zeros((grid.x.modes + grid.x.pad_width,
                                      grid.u.elements, grid.u.order,
                                      grid.v.elements, grid.v.order)) + 0j

    def compute_spectral_flux(self, distribution, field, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # self.initialize_zero_pad(grid=grid)
        self.pad_field[:-grid.x.pad_width] = field.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :] = distribution.arr_spectral
        # Pseudospectral product
        field_nodal = cp.fft.irfft(self.pad_field, norm='forward', axis=0)
        distr_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)
        nodal_flux = cp.multiply(field_nodal[:, None, None, None, None], distr_nodal)
        return cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :]

    def semi_discrete_rhs_semi_implicit(self, distribution, static_field, dynamic_field, grid):
        """ Computes the semi-discrete equation for the transport equation """
        # Compute the three fluxes with zero-padded FFTs
        self.flux_ex.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=static_field.electric_x, grid=grid)
        self.flux_ey.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=dynamic_field.electric_y, grid=grid)
        self.flux_bz.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=dynamic_field.magnetic_z, grid=grid)
        # Compute the distribution RHS
        return (grid.u.J[None, :, None, None, None] * self.u_flux(distribution=distribution, grid=grid) +
                grid.v.J[None, None, None, :, None] * self.v_flux(distribution=distribution, grid=grid))

    def semi_discrete_rhs_fully_explicit(self, distribution, static_field, dynamic_field, grid):
        """ Computes the semi-discrete equation for the transport equation """
        # Compute the three fluxes with zero-padded FFTs
        self.flux_ex.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=static_field.electric_x, grid=grid)
        self.flux_ey.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=dynamic_field.electric_y, grid=grid)
        self.flux_bz.arr_spectral = self.compute_spectral_flux(distribution=distribution,
                                                               field=dynamic_field.magnetic_z, grid=grid)
        # Compute the distribution RHS
        return (grid.u.J[None, :, None, None, None] * self.u_flux(distribution=distribution, grid=grid) +
                grid.v.J[None, :, None, None, None] * self.v_flux(distribution=distribution, grid=grid) +
                self.spectral_advection(distribution=distribution, grid=grid))

    def u_flux(self, distribution, grid):
        """ Compute the DG-projection of the u-directed flux divergence """
        # Pre-condition internal flux by the integration of the velocity coordinate
        internal = self.charge_sign * (self.flux_ex.arr_spectral + cp.einsum('rps,mijrs->mijrp',
                                                                             grid.v.translation_matrix,
                                                                             self.flux_bz.arr_spectral))
        boundary = self.charge_sign * (self.flux_ex.arr_spectral + (grid.v.device_arr[None, None, None, :, :] *
                                                                    self.flux_bz.arr_spectral))
        return (basis_product(flux=internal, basis_arr=grid.u.local_basis.internal,
                              axis=2, permutation=self.permutations[0]) -
                self.numerical_flux(distribution=distribution, flux=boundary, grid=grid, dim=0))

    def v_flux(self, distribution, grid):
        internal = self.charge_sign * (self.flux_ey.arr_spectral - cp.einsum('ijk,mikrs->mijrs',
                                                                             grid.u.translation_matrix,
                                                                             self.flux_bz.arr_spectral))
        boundary = self.charge_sign * (self.flux_ey.arr_spectral - (grid.u.device_arr[None, :, :, None, None] *
                                                                    self.flux_bz.arr_spectral))
        return (basis_product(flux=internal, basis_arr=grid.v.local_basis.internal,
                              axis=4, permutation=self.permutations[1]) -
                self.numerical_flux(distribution=distribution, flux=boundary, grid=grid, dim=1))

    def numerical_flux(self, distribution, flux, grid, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim]) + 0j

        # Set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim]) + 0j
        padded_flux[self.flux_input_slices[dim]] = flux

        # Lax-Friedrichs flux
        num_flux[self.boundary_slices[dim][0]] = -0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                                                 shift=+1, axis=self.directions[dim])[
                                                             self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][0]])
        num_flux[self.boundary_slices[dim][1]] = +0.5 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                                                 shift=-1, axis=self.directions[dim])[
                                                             self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][1]])

        # # re-use padded_flux array for padded_distribution
        padded_flux[self.flux_input_slices[dim]] = distribution.arr_spectral
        constant = cp.amax(cp.absolute(flux))

        num_flux[self.boundary_slices[dim][0]] += -0.5 * (
            cp.multiply(constant,
                        (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                 shift=+1, axis=self.directions[dim])[self.pad_slices[dim]] -
                         distribution.arr_spectral[self.boundary_slices[dim][0]]))
        )
        num_flux[self.boundary_slices[dim][1]] += -0.5 * (
            cp.multiply(constant,
                        (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                 shift=-1, axis=self.directions[dim])[self.pad_slices[dim]] -
                         distribution.arr_spectral[self.boundary_slices[dim][1]]))
        )
        return basis_product(flux=num_flux, basis_arr=grid.u.local_basis.numerical,
                             axis=self.sub_elements[dim], permutation=self.permutations[dim])

    def spectral_advection(self, distribution, grid):
        return -1j * cp.multiply(grid.x.device_wavenumbers[:, None, None, None, None],
                                 cp.einsum('ijk,mikrs->mijrs', grid.u.translation_matrix, distribution.arr_spectral))


class SpaceFlux:
    def __init__(self, resolution, c):
        self.resolution = resolution
        self.c = c

    def faraday(self, dynamic_field, grid):
        return -1j * grid.x.device_wavenumbers * dynamic_field.electric_y.arr_spectral

    def ampere(self, distribution, dynamic_field, grid):
        distribution.compute_moment_v1(grid=grid)
        return ((self.c ** 2.0) * (-1j * grid.x.device_wavenumbers * dynamic_field.magnetic_z.arr_spectral) -
                grid.charge_sign * distribution.moment_v1.arr_spectral)
