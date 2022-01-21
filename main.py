import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes
import timestep as ts

# Geometry and grid parameters
elements, order = [100, 10, 10], 10

# Grid
wavenumber = 0.1
eigenvalue = 0.508395013107024j
length = 2.0 * np.pi / wavenumber
lows = np.array([-length/2, -10, -10])
highs = np.array([length/2, 10, 10])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Variables: distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize(grid=grid, eigenvalue=eigenvalue)
# static and dynamic fields
static_fields = fields.Static(resolution=elements[0])
static_fields.gauss(distribution=distribution, grid=grid)
dynamic_fields = fields.Dynamic(resolution=elements[0])
dynamic_fields.initialize(grid=grid, eigenvalue=eigenvalue)

# Plotter: check out IC
plotter = my_plt.Plotter(grid=grid)
plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_z, y_axis='Magnetic Bz', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis='Electric Ey', spectrum=False)
plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[0, :, :, :, :], title='Mode 0')
plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[1, :, :, :, :], title='Mode 1')
plotter.show()

# plotter3d = my_plt.Plotter3D(grid=grid)
# plotter3d.distribution_contours3d(distribution=distribution, contours='adaptive', remove_average=True)
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')

# Set up fluxes
phase_space_flux = fluxes.PhaseSpaceFlux(resolutions=elements, x_modes=grid.x.modes, order=order, charge_sign=-1.0)
phase_space_flux.initialize_zero_pad(grid=grid)
space_flux = fluxes.SpaceFlux(resolution=elements[0], c=1/0.3)

# Set time-stepper
dt = 5.0e-3
steps = 5000

stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps,
                     grid=grid, phase_space_flux=phase_space_flux, space_flux=space_flux)
stepper.main_loop(distribution=distribution, static_field=static_fields, dynamic_field=dynamic_fields, grid=grid)


plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_z, y_axis='Magnetic Bz', spectrum=True)
plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis='Electric Ey', spectrum=False)
plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[0, :, :, :, :], title='Mode 0')
plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[1, :, :, :, :], title='Mode 1')
plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[2, :, :, :, :], title='Mode 2')

plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ex_energy,
                         y_axis='Electric x energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.ey_energy,
                         y_axis='Electric y energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.bz_energy,
                         y_axis='Magnetic z energy', log=True, give_rate=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=(stepper.ex_energy + stepper.ey_energy +
                                                                stepper.bz_energy + stepper.thermal_energy),
                         y_axis='Total energy', log=False)

plotter.show()
