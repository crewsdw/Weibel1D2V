import cupy as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        self.U, self.V = np.meshgrid(grid.u.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.k = grid.x.wavenumbers / grid.x.fundamental
        self.length = grid.x.length

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.flatten().get(), 'o')
        plt.xlabel('x'), plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()

        if spectrum:
            plt.figure()
            spectrum = scalar.arr_spectral.flatten().get()
            # plt.plot(self.k.flatten(), np.real(spectrum), 'ro', label='real')
            # plt.plot(self.k.flatten(), np.imag(spectrum), 'go', label='imaginary')
            plt.semilogy(self.k.flatten(), np.log(1+np.abs(spectrum)), 'o')
            plt.xlabel('Modes'), plt.ylabel(y_axis + ' spectrum')
            plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    def velocity_contourf(self, dist_slice):
        arr = np.real(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        cb = np.linspace(np.amin(arr), np.amax(arr), num=100)
        plt.figure()
        plt.contourf(self.U, self.V, arr, cb)
        plt.xlabel('u'), plt.ylabel('v'), plt.colorbar()
        plt.tight_layout()

    def velocity_contourf_complex(self, dist_slice, title='Mode 0'):
        arr_r = np.real(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        arr_i = np.imag(dist_slice.reshape(self.U.shape[0], self.U.shape[1]).get())
        arr_i[0, 0] += 1.0e-15

        cb_r = np.linspace(np.amin(arr_r), np.amax(arr_r), num=100)
        cb_i = np.linspace(np.amin(arr_i), np.amax(arr_i), num=100)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        cm = ax[0].contourf(self.U, self.V, arr_r, cb_r)
        fig.colorbar(cm, ax=ax[0])
        ax[0].set_xlabel('u'), ax[0].set_ylabel('v'), ax[0].set_title('Real')  # , ax[0].colorbar()
        cm = ax[1].contourf(self.U, self.V, arr_i, cb_i)
        ax[1].set_xlabel('u'), ax[1].set_ylabel('v'), ax[1].set_title('Imag')  # , ax[1].colorbar()
        fig.colorbar(cm, ax=ax[1])

        plt.suptitle(title), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, series_in.get() / self.length
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            exact = 2 * 0.1 * 0.508395013107024
            # 3.48694202e-01
            print('\nNumerical rate: {:0.10e}'.format(lin_fit[0]))
            # print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))  #
            print('cf. exact rate: {:0.10e}'.format(exact))
            print('The difference is {:0.10e}'.format(lin_fit[0] - exact))

    def show(self):
        plt.show()


class Plotter3D:
    """
    Plots objects on 3D piecewise (as in DG) grid
    """

    def __init__(self, grid):
        # Build structured grid, full space
        (ix, iu, iv) = (cp.ones(grid.x.elements+1),
                        cp.ones(grid.u.elements * grid.u.order),
                        cp.ones(grid.v.elements * grid.v.order))
        modified_x = 0.1 * cp.append(grid.x.device_arr, grid.x.device_arr[-1] + grid.x.dx)
        (x3, u3, v3) = (outer3(a=modified_x, b=iu, c=iv),
                        outer3(a=ix, b=grid.u.device_arr.flatten(), c=iv),
                        outer3(a=ix, b=iu, c=grid.v.device_arr.flatten()))
        self.grid = pv.StructuredGrid(x3, u3, v3)

        # build structured grid, spectral space
        ix2 = cp.ones(grid.x.modes)
        u3_2, v3_2 = (outer3(a=ix2, b=grid.u.device_arr.flatten(), c=iv),
                      outer3(a=ix2, b=iu, c=grid.v.device_arr.flatten()))
        k3 = outer3(a=grid.x.device_wavenumbers, b=iu, c=iv)
        self.spectral_grid = pv.StructuredGrid(k3, u3_2, v3_2)

    def distribution_contours3d(self, distribution, contours, remove_average=False):
        """
        plot contours of a scalar function f=f(x,y,z) on Plotter3D's grid
        """
        if remove_average:
            distribution.fourier_transform()
            distribution.arr_spectral[0, :, :, :, :] = 0
            distribution.inverse_fourier_transform()

        new_dist = np.zeros((distribution.resolutions[0]+1, distribution.resolutions[1], distribution.order,
                             distribution.resolutions[2], distribution.order))
        # append periodicity
        new_dist[:-1, :, :, :, :] = distribution.arr_nodal.get()
        new_dist[-1, :, :, :, :] = distribution.arr_nodal[0, :, :, :, :].get()

        self.grid['.'] = new_dist.reshape((new_dist.shape[0], new_dist.shape[1]*new_dist.shape[2],
                                           new_dist.shape[3]*new_dist.shape[4])).transpose().flatten()

        plot_contours = [0.1]
        if contours == 'adaptive':
            cb = np.linspace(cp.amin(distribution.arr_nodal), cp.amax(distribution.arr_nodal), num=20).tolist()
            plot_contours = self.grid.contour(cb)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True)
        # p.show_grid()
        p.show(auto_close=False)
        path = p.generate_orbital_path(n_points=36, shift=plot_contours.length)
        p.open_movie("orbit.mp4", framerate=5)
        p.orbit_on_path(path, write_frames=True)
        p.close()

    def spectral_contours3d(self, distribution, contours, option='real'):
        """
        plot contours of a scalar function f=f(k,y,z)
         on Plotter3D's spectral grid
        """
        if option == 'real':
            self.spectral_grid['.'] = np.real(distribution.spectral_flatten().get().transpose().flatten())
        if option == 'imag':
            self.spectral_grid['.'] = np.imag(distribution.spectral_flatten().get().transpose().flatten())
        if option == 'absolute':
            self.spectral_grid['.'] = np.absolute(distribution.spectral_flatten().get().transpose().flatten())
        plot_contours = self.spectral_grid.contour(contours)

        # create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True, opacity=0.75)
        p.show_grid()
        p.show()


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0).get()
