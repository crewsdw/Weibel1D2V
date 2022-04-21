import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import fluxes
import timestep as ts
import data

# Geometry and grid parameters
elements, order = [100, 22, 22], 12

# Grid
wavenumber = 0.1
# eigenvalue = 0.508395013107024j  # two-beam eigenvalue
eigenvalue = 1.2304096367176165j  # anisotropic maxwellian eigenvalue
length = 2.0 * np.pi / wavenumber
lows = np.array([-length/2, -15, -15])
highs = np.array([length/2, 15, 15])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Variables: distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize_anisotropic_maxwellian(grid=grid, eigenvalue=eigenvalue)
# static and dynamic fields
static_fields = fields.Static(resolution=elements[0])
static_fields.gauss(distribution=distribution, grid=grid)
dynamic_fields = fields.Dynamic(resolution=elements[0])
dynamic_fields.initialize(grid=grid, eigenvalue=eigenvalue)

# Plotter: check out IC
plotter = my_plt.Plotter(grid=grid)
plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_z, y_axis='Magnetic Bz', spectrum=True)
plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis='Electric Ey', spectrum=False)
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=0, title='Mode 0')
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=1, title='Mode 1')
plotter.show()

# plotter3d = my_plt.Plotter3D(grid=grid)
# plotter3d.distribution_contours3d(distribution=distribution, contours='adaptive', remove_average=True)
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')

# Set up fluxes
nu = 1.0e-1
phase_space_flux = fluxes.PhaseSpaceFlux(resolutions=elements, x_modes=grid.x.modes,
                                         order=order, charge_sign=-1.0, nu=nu)

phase_space_flux.initialize_zero_pad(grid=grid)
space_flux = fluxes.SpaceFlux(resolution=elements[0], c=1/0.3)


# Set time-stepper
dt = 2.0e-3
final_time = 101
steps = int(np.abs(final_time // dt))

# Save data
DataFile = data.Data(folder='two_stream\\', filename='april11_anisotropic')
DataFile.create_file(distribution=distribution.arr_nodal.get(),
                     density=distribution.moment0.arr_nodal.get(), current=distribution.moment_v1.arr_nodal.get(),
                     electric_x=static_fields.electric_x.arr_nodal.get(),
                     electric_y=dynamic_fields.electric_y.arr_nodal.get(),
                     magnetic=dynamic_fields.magnetic_z.arr_nodal.get())

stepper = ts.Stepper(dt=dt, resolutions=elements, order=order, steps=steps,
                     grid=grid, phase_space_flux=phase_space_flux, space_flux=space_flux)
stepper.main_loop(distribution=distribution, static_field=static_fields, dynamic_field=dynamic_fields, grid=grid,
                  data_file=DataFile)


plotter.spatial_scalar_plot(scalar=distribution.moment0, y_axis='Zero moment', spectrum=False)
plotter.spatial_scalar_plot(scalar=dynamic_fields.magnetic_z, y_axis='Magnetic Bz', spectrum=True)
plotter.spatial_scalar_plot(scalar=dynamic_fields.electric_y, y_axis='Electric Ey', spectrum=False)
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=0, title='Mode 0')
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=1, title='Mode 1')
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=2, title='Mode 2')
plotter.mode_plot_monotonic_grids(distribution=distribution, mode_idx=3, title='Mode 3')
# plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[0, :, :, :, :], title='Mode 0')
# plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[1, :, :, :, :], title='Mode 1')
# plotter.velocity_contourf_complex(dist_slice=distribution.arr_spectral[2, :, :, :, :], title='Mode 2')

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

plotter3d = my_plt.Plotter3D(grid=grid)
plotter3d.distribution_contours3d(distribution=distribution, contours='adaptive', remove_average=True)

