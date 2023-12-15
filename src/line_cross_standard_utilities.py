import numpy as np
import pylab as plt
import math

#####################################################################################################################################
#############################                    Plot functions                    ##################################################
#####################################################################################################################################

from mpl_toolkits.axes_grid1 import make_axes_locatable
def add_colorbar_subplot(fig,axes,imgs):
    for im, ax in zip(imgs,axes.flatten()):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()
    return

from matplotlib.colors import LogNorm
def plot_3D_projections(data,
                      ax=None, fig=None,
                        vmin=None,vmax=None,
                        fig_title=None,
                        log_scale=False,
                        colorbar=True,
                      cmap='plasma'):
    if cmap is None:
        cmap=my_cmap
    
    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        
    if log_scale:
        norm=LogNorm(vmin=vmin,vmax=vmax)
        vmin=None; vmax=None
    else:
        norm = None
        
    imgs = []
    for n in range(3):
        imgs.append(ax[n].matshow(data.sum(axis=n),cmap=cmap, aspect='auto',norm=norm, vmin=vmin,vmax=vmax))
    if colorbar:
        add_colorbar_subplot(fig,ax,imgs)
        
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    return

def plot_central_slices(array, 
                        log_scale=False,
                        vmin=None,vmax=None,
                        colorbar=True,
                        fig_title=None):
    
    if log_scale:
        norm=LogNorm(vmin=vmin,vmax=vmax)
        vmin=None; vmax=None
    else:
        norm = None
        
    fig,ax = plt.subplots(1,3, figsize=(12,3))
    imgs = []
    for n in range(3):
        s = [slice(None) for ii in range(3)]
        s[n] = array.shape[n]//2
        s = tuple(s)
        imgs.append(ax[n].matshow(array[s],aspect='auto', norm=norm, vmin=vmin,vmax=vmax))
    if colorbar:
        add_colorbar_subplot(fig,ax,imgs)
        
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=15)
    fig.tight_layout()
    return

#####################################################################################################################################
###################                 data pre-processing and post-processing                  ########################################
#####################################################################################################################################

def data_preprocessing(data_linear):
    maxi_rescale1 = np.max(data_linear)
    data_linear = 1e5*data_linear/maxi_rescale1
    data = np.log(data_linear+1)
    maxi_rescale2 = np.max(data)
    data = data/maxi_rescale2
    return data, maxi_rescale1, maxi_rescale2

def post_processing_prediction(prediction, maxi_rescale1, maxi_rescale2):
    prediction_linear = np.copy(prediction) * maxi_rescale2
    prediction_linear = np.exp(prediction_linear) -1
    prediction_linear = prediction_linear * maxi_rescale1/1e5
    return prediction_linear

#####################################################################################################################################
######################                    Automatically find gap shape and size                    ##################################
#####################################################################################################################################

def find_gap_shape(mask):
    '''
    Find out if the gap is a plane or a "cross" (meaning 2 planes crossing each other)
    '''
    where_mask = np.array(np.where(mask))
    if np.all( (np.array(mask.shape)-1) == (np.max(where_mask, axis=1) - np.min(where_mask, axis=1)) ):
        shape = 'cross'
    else:
        shape = 'line'
    return shape

def find_line_gap_perpendicular_axis(mask):
    where_mask = np.array(np.where(mask))
    bool_array = (np.array(mask.shape)-1) == (np.max(where_mask, axis=1) - np.min(where_mask, axis=1))
    axis_perpendicular = np.where(~bool_array)[0][0]
    return axis_perpendicular

def find_line_gap_size_position(mask, axis_perpendicular):
    where_gap = np.unique(np.where(mask)[axis_perpendicular])
    gap_size = len(where_gap)
    mask_position = np.mean(where_gap)
    return gap_size, mask_position

def find_cross_gap_parallel_axis(mask):
    # Find parallel vector if gap is a cross
    matrix = np.zeros((3,3))
    for axis in range(3):
        s = [slice(None) for n in range(3)]
        s[axis] = slice(0,1)
        s = tuple(s)
        where_mask = np.array(np.where(mask[s])) # Mask shouldn't touch the border of the array !!!!
        matrix[axis] += np.max(where_mask, axis=1) - np.min(where_mask, axis=1)
    bool_matrix = ~(np.isin(matrix, (np.array(mask.shape)-1)) + (matrix==0))
    axis_parallel = np.delete(np.arange(3), np.unique(np.where(bool_matrix)))[0]
    return axis_parallel

def find_cross_gap_size_position(mask, axis_parallel):
    axis_list = np.delete(np.arange(3), axis_parallel)

    gap_list = np.zeros(3)
    mask_position = np.zeros(3)
    pixels_masked = [[],[],[]]

    gap_list[axis_parallel] = np.nan
    mask_position[axis_parallel] = np.nan

    for axis in axis_list:
        s = [slice(None) for n in range(3)]
        s[axis] = slice(0,1)
        s = tuple(s)
        where_mask = np.array(np.where(mask[s]))
        axis_temp = np.delete(np.arange(3), [axis_parallel,axis])
        where_gap = np.unique(where_mask[axis_temp])
        gap_list[axis] = len(where_gap)
        mask_position[axis] = np.mean(where_gap)
        pixels_masked[axis] = where_gap
    mask_position[axis_list] = mask_position[axis_list[::-1]] # The positions were inverted
    pixels_masked = np.array(pixels_masked, dtype=object)
    pixels_masked[axis_list] = pixels_masked[axis_list[::-1]]

    if len(np.unique(gap_list[~np.isnan(gap_list)])) !=1:
        print('Error, gap size is not the same for both plane. Program not ready for this possibility yet.')
    gap_size = int(np.unique(gap_list[~np.isnan(gap_list)])[0])
    
    return gap_size, mask_position, pixels_masked

def find_gap_parameters(mask,
                        verbose=False):
    gap_params = {}
    gap_params['gap_shape'] = find_gap_shape(mask)
    
    if gap_params['gap_shape'] == 'line' :
        gap_params['axis_perpendicular'] = find_line_gap_perpendicular_axis(mask)
        gap_params['gap_size'], gap_params['mask_position'] = find_line_gap_size_position(mask, 
                                                                                               gap_params['axis_perpendicular'])
        gap_params['mask_position'] = int(math.floor(gap_params['mask_position']))
        
    if gap_params['gap_shape'] == 'cross':
        gap_params['axis_parallel'] = find_cross_gap_parallel_axis(mask)
        gap_params['gap_size'], gap_params['mask_position'],  gap_params['pixel_masked'] = find_cross_gap_size_position(mask,
                                                                                                gap_params['axis_parallel'])
        for n in range(3):
            if ~np.isnan(gap_params['mask_position'][n]):
                gap_params['mask_position'][n] = int(math.floor(gap_params['mask_position'][n]))
    if verbose:
        for key in gap_params.keys():
            print('{} : {}'.format(key, gap_params[key]))
    
    return gap_params



#####################################################################################################################################
######################               Gap creations (noly to create fake gapped data)               ##################################
#####################################################################################################################################

def add_line_gap_random_position3D(data, 
                                    axis_perpendicular=1, gap_size = None, gap_position = None, min_distance_from_center = 0): 

    
    mask = np.zeros(data.shape)
        
    if gap_position is None:
        center_position = data.shape[axis_perpendicular] / 2.
        random_list = np.arange(-gap_size//2, int(round(center_position-min_distance_from_center)))
        random_list = np.concatenate((random_list, 
                              np.arange(int(round(center_position+min_distance_from_center)),
                                        data.shape[axis_perpendicular]+gap_size//2)))
        gap_position = np.random.choice(random_list)

        
    x,y,z = np.indices(data.shape)
    gap_start = gap_position - gap_size // 2
    gap_end = gap_position + gap_size // 2

    gap_start = gap_position - gap_size // 2
    gap_end = gap_position + gap_size // 2
    if gap_size % 2 > 0: # se e' dispari
        gap_end += 1

    s = [slice(None) for n in range(data.ndim)]
    s[axis_perpendicular] = slice(gap_start, gap_end)
    s = tuple(s)
    mask[s] = 1
        
    return mask

def add_cross_gap_random_position3D(data, 
                                    axis_parallel=2, gap_size = None, gap_position_list = [None,None],
                                    min_distance_from_center = 0): 
    
    axis_list = np.delete(np.arange(3), axis_parallel)
    
    mask = np.zeros(data.shape)
    
    for ii, axis in enumerate(axis_list):
        mask +=  add_line_gap_random_position3D(data, 
                                                axis_perpendicular=axis, gap_size = gap_size,
                                                gap_position = gap_position_list[ii],
                                                min_distance_from_center = min_distance_from_center)
        
    mask[mask!=0] = 1
        
    return mask
