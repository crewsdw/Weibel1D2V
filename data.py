import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self, distribution, density, current, electric_x, electric_y, magnetic):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1], distribution.shape[2],
                                       distribution.shape[3], distribution.shape[4]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density]),
                             chunks=True,
                             maxshape=(None, density.shape[0]),
                             dtype='f')
            f.create_dataset('current', data=np.array([current]),
                             chunks=True,
                             maxshape=(None, current.shape[0]),
                             dtype='f')
            f.create_dataset('electric_x', data=np.array([electric_x]),
                             chunks=True,
                             maxshape=(None, electric_x.shape[0]),
                             dtype='f')
            f.create_dataset('electric_y', data=np.array([electric_y]),
                             chunks=True,
                             maxshape=(None, electric_y.shape[0]),
                             dtype='f')
            f.create_dataset('magnetic', data=np.array([magnetic]),
                             chunks=True,
                             maxshape=(None, magnetic.shape[0]),
                             dtype='f')
            f.create_dataset('time', data=[0.0], chunks=True, maxshape=(None,))
            f.create_dataset('total_energy', data=[], chunks=True, maxshape=(None,))
            f.create_dataset('total_density', data=[], chunks=True, maxshape=(None,))

    def save_data(self, distribution, density, current, electric_x, electric_y, magnetic, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['current'].resize((f['current'].shape[0] + 1), axis=0)
            f['electric_x'].resize((f['electric_x'].shape[0] + 1), axis=0)
            f['electric_y'].resize((f['electric_y'].shape[0] + 1), axis=0)
            f['magnetic'].resize((f['magnetic'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['density'][-1] = density
            f['current'][-1] = current
            f['electric_x'][-1] = electric_x
            f['electric_y'][-1] = electric_y
            f['magnetic'][-1] = magnetic
            f['time'][-1] = time

    def save_inventories(self, total_energy, total_density):
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['total_energy'].resize((f['total_energy'].shape[0] + 1), axis=0)
            f['total_density'].resize((f['total_density'].shape[0] + 1), axis=0)
            # Save data
            f['total_energy'][-1] = total_energy.get()
            f['total_density'][-1] = total_density.get()

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            pdf = f['pdf'][()]
            density = f['density'][()]
            current = f['current'][()]
            electric_x = f['electric_x'][()]
            electric_y = f['electric_y'][()]
            magnetic = f['magnetic'][()]
            # total_eng = f['total_energy'][()]
            # total_den = f['total_density'][()]
        return time, pdf, density, current, electric_x, electric_y, magnetic
