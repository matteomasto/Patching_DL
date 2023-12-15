# AUTHOR: Dr. Ewen Bellec

# Instead of loading all data at once, I make this data generator.
# During training, it will load only one batch_size number of data at a time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pylab as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

######################################################################################################################################
################################                  Data generator                                ######################################
######################################################################################################################################


    
######################################################################################################################################
################################                  Load dataset files                            ######################################
######################################################################################################################################
    
import random
def dataset_files(path,
                  shuffle=False,
                  nb_files = None,
                  verbose=True):
    files = os.listdir(path)
    files = [path+f for f in files]

    if shuffle :
        random.shuffle(files)

    if verbose:
        print('nb files :',len(files))

    if nb_files is not None:
        files = files[:nb_files]
        if verbose:
            print('nb files kept :',len(files))
    return files
    
def split_dataset_files(files,
                        validation_ratio = .1, validation_max_size = 500,
                        test_ratio = .05, test_max_size = 500,
                        verbose=True):
    nb_test_files = min(int(len(files)*test_ratio) , test_max_size)
    nb_validation_files = min(int(len(files)*validation_ratio) , validation_max_size)

    test_files = files[:nb_test_files] # take 500 of the files for a test dataset
    val_files = files[nb_test_files : nb_test_files + nb_validation_files] # take 500 of the files for a validation dataset
    train_files = files[nb_test_files + nb_validation_files:] # The rest of the files will be in the training dataset

    if verbose:
        print('training dataset elements :', len(train_files))
        print('validation dataset elements :',len(val_files))
        print('test dataset elements :',len(test_files))
    
    return train_files, val_files, test_files
    
######################################################################################################################################
################################                       Utilities                                ######################################
######################################################################################################################################
   

# Just a function for plots at the end
def crop_array_half_size(array):
    '''
    Works for any dimension
    '''
    shape = array.shape
    s = [slice(shape[n]//2-shape[n]//4, shape[n]//2+shape[n]//4) for n in range(array.ndim)]
    return array[tuple(s)]

def custom_sigmoid(x, 
                   amplitude=15., slope=1.):
    return amplitude*(2*(1./(1+tf.math.exp(-slope*x))-.5))

from numpy.fft import fftshift, ifftshift, ifftn, fftn
def reconstruct_object(diffracted_module, 
                       diffracted_phase):
    
    Fexp = diffracted_module*np.exp(1.0j*diffracted_phase)
    return ifftshift(ifftn(fftshift(Fexp)))

def check_path_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

######################################################################################################################################
################################                     Plot utilities                             ######################################
######################################################################################################################################


def plot_object(obj,
                unwrap=True,
                crop=False,
                fig=None, ax=None):
    
    module= np.abs(obj)
    
    if unwrap:
        phase = unwrap_phase(obj)
    else:
        phase = np.angle(obj)
    phase[module<.3*np.max(module)] = np.nan # Just to clean the plot
    
    if crop:
        module = crop_array_half_size(module)
        phase = crop_array_half_size(phase)
    
    if fig is None:
        fig,ax = plt.subplots(1,2, figsize=(8,4))
        
    im0 = ax[0].matshow(module, cmap='gray_r')
    ax[0].set_title('object module', fontsize=20)
    im1 = ax[1].matshow(phase, cmap='hsv')
    ax[1].set_title('object phase', fontsize=20)

    for axe, img in zip(ax,[im0,im1]):
        divider = make_axes_locatable(axe)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')

    fig.tight_layout()
    
    return