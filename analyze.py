import numpy as np
import grid as g
import variables as var
import fields
import plotter as my_plt
import data
import cupy as cp

# Geometry and grid parameters
elements, order = [64, 20, 20], 12

# Grid
wavenumber = 0.1
eigenvalue = 0.508395013107024j
length = 2.0 * np.pi / wavenumber
lows = np.array([-length/2, -12, -12])
highs = np.array([length/2, 12, 12])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, charge_sign=-1.0)

# Read data
data_file = data.Data(folder='two_stream\\', filename='test_jan26')
time_data, f_data, n_data, v_data, ex_data, ey_data, b_data = data_file.read_file()

# Set up plotter
# Plotter = my_plt.Plotter(grid=grid)
# Plotter.plot_many_scalars(times=time_data, scalars=n_data, y_axis='Density', save_name='density')
# Plotter.plot_many_scalars(times=time_data, scalars=v_data, y_axis='Mean velocity', save_name='velocity')
# Plotter.plot_many_scalars(times=time_data, scalars=ex_data, y_axis='x-Electric field', save_name='electric_x')
# Plotter.plot_many_scalars(times=time_data, scalars=ey_data, y_axis='y-Electric field', save_name='electric_y')
# Plotter.plot_many_scalars(times=time_data, scalars=b_data, y_axis='Magnetic field', save_name='magnetic')
# Plotter.show()

# Look at final distribution
final_distribution = var.Distribution(resolutions=elements, order=order)
final_distribution.arr_nodal = cp.asarray(f_data[-1, :, :, :, :])
final_distribution.fourier_transform()

plotter3d = my_plt.Plotter3D(grid=grid)
plotter3d.distribution_contours3d(distribution=final_distribution, contours='adaptive', remove_average=False)

# Loop through time data
# for idx, time in enumerate(time_data):

