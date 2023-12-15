import numpy as np
from scipy.interpolate import RegularGridInterpolator

def create_interpolation_with_gap(data_masked, mask,
                                 interpolation_method = 'linear',
                                 verbose=True):
    masked_positions = np.unique(np.where(mask==1)[1])
    if verbose:
        print('masked pixels along axis 1 :', masked_positions)
    start = np.min(masked_positions)
    end = np.max(masked_positions)
    size = end-start + 1
    
    x = np.arange(data_masked.shape[0])
    y = np.concatenate((np.arange(start), 
                        np.arange(end+1,data_masked.shape[1])))
    # For the second axis I don't take any pixels inside the gap.
    z = np.arange(data_masked.shape[2])

    # I create the data array with the gap removed (not putting 0s but removed !)
    data_without_gap = np.zeros((data_masked.shape[0], data_masked.shape[1]-size, data_masked.shape[2]))
    data_without_gap[:,:start] += data_masked[:,:start]
    data_without_gap[:,start:] += data_masked[:,end+1:]

    interp = RegularGridInterpolator((x, y, z), data_without_gap,
                                    method=interpolation_method) # My interpolation function
    
    return interp

def interpolate_gap(data_masked, mask,
                    interpolation_method = 'linear'):
    
    # My interpolation function
    interp = create_interpolation_with_gap(data_masked, mask,
                                     interpolation_method = interpolation_method)
    
    # Create the grid points
    x,y,z = np.indices(data_masked.shape)

    # Bad coding, didn't manage to make it in 1 line
    points = np.zeros(x.shape + (3,))
    points[...,0] += x
    points[...,1] += y
    points[...,2] += z
    
    data_inter = interp(points)
    data_inter = np.reshape(data_inter, x.shape)
    return data_inter

from scipy.interpolate import griddata
def create_interpolated_data(data,mask):
    nearest = np.zeros(data.shape)
    linear = np.zeros(data.shape)
    cubic = np.zeros(data.shape)

    for i in range(data.shape[0]):
        print(data.shape[0]-i,end=' ')

        x,y= np.indices((data.shape[0],data.shape[0]))

        x = x[mask[:,:,i]==0]
        y = y[mask[:,:,i]==0]
        values = np.copy(data[:,:,i][mask[:,:,i]==0])

        points = np.zeros((len(x), 2))
        points[:,0] += x
        points[:,1] += y

        grid_x, grid_y = np.indices((data.shape[0],data.shape[0]))  

        nearest[:,:,i] = interp_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
        linear[:,:,i] = interp_nearest = griddata(points, values, (grid_x, grid_y), method='linear')
        cubic[:,:,i] = interp_nearest = griddata(points, values, (grid_x, grid_y), method='cubic')
    return nearest, linear, cubic