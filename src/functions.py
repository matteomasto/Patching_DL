# AUTHORS: Matteo Masto and Dr. Ewen Bellec

# this is the list of function used in the main code for 3D inpainting

import numpy as np
from numpy.fft import fftshift, ifftshift
import pylab as plt
import tensorflow as tf 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorflow.keras.layers import Conv3D, LeakyReLU, Conv3DTranspose, Add, BatchNormalization, Input, MaxPool3D, UpSampling3D, Concatenate
from tensorflow.keras import Model

plt.rc('image', cmap='plasma')


def accuracy(gt, pred, mask):
    
       
    gap_true = gt*mask
    gap_pred = pred*mask
    
    mae = np.mean(np.abs(gap_true-gap_pred))
    max_mae = np.mean(np.abs(gap_true))

    acc = (1.0 - mae/max_mae)*100

    if max_mae == 0:
        
        if mae == 0:
            acc = 100
        else:
            acc = np.nan
        
    
    if acc<0:
        acc = 0
    
    return acc

def average_scatter(x,y,N,
                    plot=False, verbose=False):
    dx = (np.max(x) - np.min(x)) / N
    x1d = np.linspace(np.min(x), np.max(x)-dx, N) + dx/2.
    average= np.zeros(N)
    for n in range(N):
        if verbose:
            print(n,end=' ')
        indices = np.abs(x-x1d[n]) <= dx
        average[n] += np.mean(y[indices])
         
    if plot:
        plt.figure()
        plt.plot(x,y,'.')
        plt.plot(x1d,average, 'o-')
        
    return x1d, average 

def plot_3D_projections(data,
                      ax=None, fig=None,
                        fig_title=None,
                      cmap='plasma'):
    
    "function to plot slices of 3D array"
    
    if cmap is None:
        cmap=my_cmap
    
    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        
    for n in range(3):
        ax[n].matshow(data.sum(axis=n),cmap=cmap, aspect='auto')
        
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    return

def split_dataset_files(files,
                        validation_ratio = .1, validation_max_size = 1000,
                        test_ratio = .05, test_max_size = 800,
                        verbose=True):
    nb_test_files = min(int(len(files)*test_ratio) , test_max_size)
    nb_validation_files = min(int(len(files)*validation_ratio) , validation_max_size)
    
    "this function splits files into 3 datasets: training, validation and test"

    test_files = files[:nb_test_files] # take 500 of the files for a test dataset
    val_files = files[nb_test_files : nb_test_files + nb_validation_files] # take 500 of the files for a validation dataset
    train_files = files[nb_test_files + nb_validation_files:] # The rest of the files will be in the training dataset

    if verbose:
        print('training dataset elements :', len(train_files))
        print('validation dataset elements :',len(val_files))
        print('test dataset elements :',len(test_files))
    
    return train_files, val_files, test_files

def add_vertical_gap3D(Fexp, gap_size = None, gap_position = None, min_distance_from_center = 20, plot=False):
    
    "this function creates vertical plane gap at a given position, for a given gap size, in a 3D array"
    
    if gap_size is None:
        gap_size = np.random.randint(8,9)

    off = 20
    center_position = Fexp.shape[1] / 2.
    random_list = np.arange(-gap_size//2, int(round(center_position-min_distance_from_center)))
    random_list = np.concatenate((random_list, 
                          np.arange(int(round(center_position+min_distance_from_center)), Fexp.shape[1]+gap_size//2)))
    
    if gap_position is None:
        gap_position = np.random.choice(random_list)

    mask = np.zeros(Fexp.shape)
    x,y,z = np.indices(Fexp.shape)

    gap_start = gap_position - gap_size // 2
    gap_end = gap_position + gap_size // 2
    if gap_size % 2 > 0: # se e' dispari
        gap_end += 1
        
    mask[:, gap_start:gap_end, :] = 1 
        
    if plot:
        plot_3D_projections(Fexp*(1.-mask))       
        
    return mask


def add_cross_gap3D_portions(Fexp, gap_size = None, gap_position = None, min_distance_from_center = 0, plot = False):
    
    "This function creates a cross-shaped gap in a 3D array"                                   
    if gap_size is None:
        gap_size = np.random.randint(8,9)
        
    
    off = 20
    center_position = Fexp.shape[1] / 2.
    random_list = np.arange(-gap_size//2, int(round(center_position-min_distance_from_center)))
    random_list = np.concatenate((random_list, 
                          np.arange(int(round(center_position+min_distance_from_center)), Fexp.shape[1]+gap_size//2)))
    
    if gap_position is None:
        gap_position = np.random.choice(random_list)

    mask = np.zeros(Fexp.shape)
    x,y,z = np.indices(Fexp.shape)
    
    gap_start = gap_position - gap_size // 2
    gap_end = gap_position + gap_size // 2
    
    if gap_size % 2 > 0: # is odd
        gap_end += 1
        
    mask[:, gap_start:gap_end, :] = 1 


    rot = np.random.choice([1,3])
    
    mask = np.rot90(mask,1)
    mask2= add_vertical_gap3D(Fexp,gap_size = gap_size, gap_position = 16)
    mask1 = mask + mask2
    mask1[mask1 == 2 ] = 1
    
    if plot:
        plot_3D_projections(Fexp*(1.-mask1)) 
    
    return mask1

def add_cross_gap3D(Fexp, gap_size = None, gap_position = None, min_distance_from_center = 0, plot = False):
    
    "This function creates a cross-shaped gap in a 3D array"                                   
    if gap_size is None:
        gap_size = np.random.randint(8,9)
        
    
    off = 20
    center_position = Fexp.shape[1] / 2.
    random_list = np.arange(-gap_size//2, int(round(center_position-min_distance_from_center)))
    random_list = np.concatenate((random_list, 
                          np.arange(int(round(center_position+min_distance_from_center)), Fexp.shape[1]+gap_size//2)))
    
    if gap_position is None:
        gap_position = np.random.choice(random_list)

    mask = np.zeros(Fexp.shape)
    x,y,z = np.indices(Fexp.shape)
    
    gap_start = gap_position - gap_size // 2
    gap_end = gap_position + gap_size // 2
    
    if gap_size % 2 > 0: # is odd
        gap_end += 1
        
    mask[:, gap_start:gap_end, :] = 1 


    rot = np.random.choice([1,3])
    
    mask = np.rot90(mask,1)
    mask2= add_vertical_gap3D(Fexp,gap_size = gap_size, gap_position = gap_position)
    mask1 = mask + mask2
    mask1[mask1 == 2 ] = 1
    
    if plot:
        plot_3D_projections(Fexp*(1.-mask1)) 
    
    return mask1


from tensorflow import keras
from tensorflow.keras import utils

from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar_subplot(fig,axes,imgs,
                         nbins = None):
    for im, ax in zip(imgs,axes.flatten()):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        if nbins is not None:
            cbar.ax.locator_params(nbins=nbins)
    fig.tight_layout()
    return


class DataGeneratorDiffraction(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1, shuffle=True,
                 normalize = False, input_log_data=False, gap_size = None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.input_log_data = input_log_data
        self.gap_size = gap_size
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, 2))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            if '.npz' in ID:
                data = np.load(ID)

                if 'Inorm' in data.files:                    # loading sim data
                    I = data['Inorm']
                    I = np.interp(I, (I.min(),I.max()), (0,1))
                    
                    choice = np.random.choice(5)
                    
                    if choice < 4:
                        mask = add_vertical_gap3D(I, gap_size = self.gap_size, gap_position = 16, min_distance_from_center = 0)

                    else:     
                        mask = add_cross_gap3D_portions(I, gap_size = self.gap_size, gap_position = None, min_distance_from_center = 0)

                    X[i,...,0] = I * (1.-mask)

                    y[i,...,0] = I
                    y[i,...,1] = mask            
            

        return tf.cast(X, tf.float32), tf.cast(y, tf.float32)
    
def create_datagenerators(train_files, val_files, test_files,
                          batch_size = 32, gap_size = None,
                          normalize = False, input_log_data = False, 
                          shuffle_test=False):
    
    "create data generators for each dataset"
    
    if gap_size is None:
            gap_size = np.random.randint(1,16)
    
    train_gen = DataGeneratorDiffraction(train_files, batch_size=batch_size,
                                               input_log_data=input_log_data, gap_size = gap_size)
    
    val_gen = DataGeneratorDiffraction(val_files, batch_size=batch_size,
                                              input_log_data=input_log_data, gap_size = gap_size)
    
    test_gen = DataGeneratorDiffraction(test_files, shuffle=shuffle_test, batch_size=batch_size,
                                               input_log_data=input_log_data, gap_size = gap_size)
    
    return train_gen, val_gen, test_gen


def custom_sigmoid(x):
    return 1./(1+tf.math.exp(-1.5*x))

def encoder_block(x_input, num_filters, ker):

    x = Conv3D(num_filters, ker, strides=1, padding="same")(x_input)
    x = MaxPool3D(2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    s = tf.identity(x)

    return x, s

def encoder_block_mod(x_input,ker, num_filters, rate):
    
    reg = num_filters//4
        
    x1 = Conv3D(reg, ker, strides=1, dilation_rate = rate[0], padding="same")(x_input)
    x2 = Conv3D(reg, ker, strides=1, dilation_rate = rate[1], padding="same")(x_input)
    x3 = Conv3D(reg, ker, strides=1, dilation_rate = rate[2], padding="same")(x_input)
    x4 = Conv3D(reg, ker, strides=1, dilation_rate = rate[3], padding="same")(x_input)
    x = tf.concat([x_input,x1,x2,x3,x4], axis = -1)
    x = MaxPool3D(2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    s = tf.identity(x)

    return x, s

def decoder_block(x_input, skip_input, num_filters, ker):
    
    x = Concatenate()([x_input, skip_input])
    x = Conv3DTranspose(num_filters, ker, strides=1, padding="same")(x)
    x = UpSampling3D(2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x

def build_unet_mod(input_shape):
    inputs = Input(input_shape)                # 32x32 

    x, s1 = encoder_block_mod(inputs,3,32,[4,3,2,1])        # 16x16x32

    x, s2 = encoder_block_mod(x,2, 64,[3,2,2,1])             # 8x8x96
 
    x, s3 = encoder_block(x,128,2)            # 4x4x222

    x, s4 = encoder_block(x, 256, 2)            # 2x2x478

    
    x = Conv3DTranspose(478, 2, strides=1, padding="same")(x)
    x = UpSampling3D(2)(x)
    x = LeakyReLU(alpha = 0.2)(x)  #4x4
   
    x = decoder_block(x,s3, 222,2)  #8x8
    
    x = decoder_block(x,s2, 96,2)  #16x16
   
    x = decoder_block(x,s1, 32,2)  #32x32

    x = Conv3D(24,4, strides=1, padding='same')(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv3D(12,3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x) 
    
    x = Conv3D(6,2, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x) 

    outputs = Conv3D(1,1, padding="same", activation=custom_sigmoid)(x)
    
    
    model = Model(inputs, outputs, name="U-Net")
    
    return model

def ManualGradient(gap):
    ' manual calculation of the gradient in the gap region'
    dgap_xx = []
    dgap_yy = []
    dgap_zz = []
    dgap_xy = []
    dgap_xz = []
    dgap_yz = []
    
    

    for i in range(32-1):
        
        dgap_xx.append(gap[:,i+1,:,:] - gap[:,i,:,:])
        dgap_yy.append(gap[:,:,i+1,:] - gap[:,:,i,:])
        dgap_zz.append(gap[:,:,:,i+1] - gap[:,:,:,i])
        dgap_xy.append(gap[:,i+1,:,:] - gap[:,:,i,:]) 
        dgap_xz.append(gap[:,i+1,:,:] - gap[:,:,:,i])
        dgap_yz.append(gap[:,:,i+1,:] - gap[:,:,:,i])  
        
        
    dgap_xx = tf.stack(dgap_xx) 
    dgap_yy = tf.stack(dgap_yy)
    dgap_zz = tf.stack(dgap_zz)
    dgap_xy = tf.stack(dgap_xy)
    dgap_xz = tf.stack(dgap_xz)
    dgap_yz = tf.stack(dgap_yz)
        
    return [dgap_xx, dgap_yy, dgap_zz, dgap_xy, dgap_xz, dgap_yz]

# LOSS FUNCTION______________________________________________________________________________________________________________________

def MyLoss_noisy(y_true, y_pred):
    
    mask = y_true[...,1]
    
    y_pred_gap = y_pred[...,0]*mask  # isolate the gap, the rest of the image is zero
    y_true_gap = y_true[...,0]*mask
     
    mod_pred1 = y_pred_gap + y_true[...,0]*(1.-mask)   # add the g.t. image to the predicted gap
    L1 = tf.reduce_mean(tf.math.abs(y_true_gap - y_pred_gap), axis = [1,2,3])
    
#__________________________________________________________________________________________________________

    L_corr = 0
    for i in range(32):
        
        y_truei = tf.reshape(y_true[:,i,...,0], (32,32,32,1))
        y_predi = tf.reshape(mod_pred1[:,i,...], (32,32,32,1))
    
        y_true1 = tf.concat([y_truei, tf.zeros((32,32,32,2))], -1)
        y_pred1 = tf.concat([y_predi, tf.zeros((32,32,32,2))], -1)

        L_corr += 1 - tf.image.ssim(y_true1,y_pred1, 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        
#__________________________________________________________________________________________________________

#  loss on the gradient in the mask region
        # Gradient
    
    pred_mask1 = tf.reshape(y_true[...,0], (32,32,32,32,1))
    true_mask1 = tf.reshape(mod_pred1, (32,32,32,32,1))
    
    grad_gap_true = ManualGradient(true_mask1)
    grad_gap_pred = ManualGradient(pred_mask1)
    
    L2_0 = tf.reduce_mean(tf.math.abs(grad_gap_true[0] - grad_gap_pred[0]), axis = [0,2,3])
    L2_1 = tf.reduce_mean(tf.math.abs(grad_gap_true[1] - grad_gap_pred[1]), axis = [0,2,3])
    L2_2 = tf.reduce_mean(tf.math.abs(grad_gap_true[2] - grad_gap_pred[2]), axis = [0,2,3])
    L2_3 = tf.reduce_mean(tf.math.abs(grad_gap_true[3] - grad_gap_pred[3]), axis = [0,2,3])
    L2_4 = tf.reduce_mean(tf.math.abs(grad_gap_true[4] - grad_gap_pred[4]), axis = [0,2,3])
    L2_5 = tf.reduce_mean(tf.math.abs(grad_gap_true[5] - grad_gap_pred[5]), axis = [0,2,3])
    
    L_grad = L2_0+L2_1+L2_2+L2_3+L2_4+L2_5
    

    return  L1 + L_grad + L_corr 



import os
def PickRandomDiffraction(path, gap_size, plot = True):
    filesall = os.listdir(path)
    len(filesall)

    index = np.random.randint(0,len(filesall))
    data = np.load(path+filesall[index])

    if 'I' in data.files:    # loading exp data
        Ii = 1e5*data['I']          
        I = np.log(Ii+1)
        I = I/np.max(I)

        mask = add_cross_gap_3D(I, gap_size = gap_size, gap_position = None ,min_distance_from_center = 0) 

        data_masked = I * (1.-mask)


    elif 'Fexp' in data.files:                    # loading sim data
        Fexp = data['Fexp']
        Ii = np.abs(Fexp)**2.
        I = np.log(Ii+1) 
        I = I/np.max(I)

        if I[64,64,64] <= I[120,120,120]:
            I = ifftshift(I)

        mask = add_cross_gap_3D(I, gap_size = gap_size, gap_position = None, min_distance_from_center = 0) 

        data_masked = I * (1.-mask)
    
    if plot is True:
        plt.matshow(data_masked[:,:,data_masked.shape[0]//2])
        
    return I, data_masked, mask


def SmartCrop3D_mask(I,out_size, mode = 'normal', skip = None, nb_images = None, probabilities = None, matrix = None):
        
    "This function crops sub-portions of an entire 3D array. These nb_images portions, of size out_size are returned in an array with        shape (nb_images, out_size, out_size, out_size)"
        
    inp_size = I.shape
    div = inp_size[0]//out_size  
    
    if mode == 'normal':  #divide without overlaps
        
        a = np.split(I,div,axis=0) 
        b = np.zeros((int(div*div*div),out_size, out_size, out_size))  
        
        for i in range(len(a)):
            for j in range(len(a)):
                for k in range(len(a)):
            
                   b[i*div*div+j*div+k] = np.split(np.split(a[i],div,axis=1)[j], div, axis = 2)[k] #[]
                
        return b
    
    if mode == 'overlap':  # overlap
        
        
        b = np.zeros((((inp_size[0]-out_size)//skip)**3, out_size, out_size, out_size))
        k = 0
        
        for i in range((inp_size[0]-out_size)//skip):
            for j in range((inp_size[0]-out_size)//skip):
                for t in range((inp_size[0]-out_size)//skip):
                    
                    b[k] = I[skip*i:skip*i+out_size , skip*j:skip*j+out_size, skip*t:skip*t+out_size]
                    k += 1 
                    
                
        return b
    
    if mode == 'random':  # overlap in the external region
        
        b = np.zeros((nb_images, out_size,out_size,out_size))
        
        for i in range(nb_images):
            x = np.random.randint(out_size//2+1, inp_size[0] - out_size//2)
            y = np.random.randint(out_size//2, inp_size[1] - out_size//2)
            z = np.random.randint(out_size//2, inp_size[2] - out_size//2)
            b[i] = I[x-out_size//2:x+out_size//2,y-out_size//2:y+out_size//2,z-out_size//2:z+out_size//2]
                
        return b
    
    if mode == 'custom':
        
        b = np.zeros((matrix.shape[1], out_size,out_size,out_size))
                     
        for i in range(matrix.shape[1]):
                     
            x = matrix[0,i]
            y = matrix[1,i]
            z = matrix[2,i]
            b[i] = I[x-out_size//2:x+out_size//2,y-out_size//2:y+out_size//2,z-out_size//2:z+out_size//2]
                
        return b
    
    
class DataGeneratorDiffraction_ft(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_images, batch_size=32, dim=(32,32,32), n_channels=1, shuffle=True,
                 normalize=False, input_log_data=False, gap_size=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_images = list_images  # List of images
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.input_log_data = input_log_data
        self.gap_size = gap_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_images = self.list_images[start_index:end_index]

        # Generate data
        X, y = self.__data_generation(batch_images)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.list_images)

    def __data_generation(self, batch_images):
        'Generates data containing batch_size samples'
        # Initialization
        batch_size = len(batch_images)
        X = np.empty((batch_size, *self.dim, 1))
        y = np.empty((batch_size, *self.dim, 2))

        # Generate data
        for i, I in enumerate(batch_images):
            # Process your image data as needed here
            
            I = np.interp(I, (I.min(),I.max()), (0,1))
                    
            choice = np.random.choice(5)

            if choice < 4:
                mask = add_vertical_gap3D(I, gap_size = self.gap_size, gap_position = 16, min_distance_from_center = 0)

            else:     
                mask = add_cross_gap3D_portions(I, gap_size = self.gap_size, gap_position = None, min_distance_from_center = 0)

            X[i,...,0] = I * (1.-mask)

            y[i,...,0] = I
            y[i,...,1] = mask   
            pass

        return tf.cast(X, tf.float32), tf.cast(y, tf.float32)
    

def create_datagenerator_ft(images, batch_size=32, gap_size=None, normalize=False, input_log_data=False):
    data_gen = DataGeneratorDiffraction_ft(images, batch_size=batch_size, input_log_data=input_log_data, gap_size=gap_size)
    return data_gen


def create_target_dataset(data, mask, nb_images):
    mask_nan = np.zeros(mask.shape)
    mask_nan[mask == 1] = np.nan
#     nb_images = int(32*200)
    i = 0
    output = np.zeros((nb_images,32,32,32))

    Ilog = np.log(data+1)
    Inorm = np.interp(Ilog, (Ilog.min(),Ilog.max()), (0,1))
    I = Inorm*(1.-mask_nan)

    out_size = 32

    scaled_intensity_distribution = ((Inorm / np.sum(Inorm)) / np.sum(Inorm / np.sum(Inorm)))

    distrib_x = scaled_intensity_distribution.mean(axis = (1,2))
    distrib_x = distrib_x[16:-16]
    distrib_x /= np.sum(distrib_x)


    distrib_y = scaled_intensity_distribution.mean(axis = (0,2))
    distrib_y = distrib_y[16:-16]
    distrib_y /= np.sum(distrib_y)


    distrib_z = scaled_intensity_distribution.mean(axis = (0,1))
    distrib_z = distrib_z[16:-16]
    distrib_z /= np.sum(distrib_z)

    while(i < nb_images):  # finche non ho tutte le immagini

        x = np.random.choice(np.arange(16,len(distrib_x)+16,1), p=distrib_x)
        y = np.random.choice(np.arange(16,len(distrib_y)+16,1), p=distrib_y)
        z = np.random.choice(np.arange(16,len(distrib_z)+16,1), p=distrib_z)
    #     x,y,z = sample_3d_coordinates_with_gaussian(Inorm,scale_factor = 100)
        portion = I[x-out_size//2:x+out_size//2,y-out_size//2:y+out_size//2,z-out_size//2:z+out_size//2]  #ritaglio la porzione                  

        if np.any(np.isnan(portion)) == False:
            output[i] = portion
            i +=1
        else:
            continue
            
    return output

def scatter2d_to_image(x,y,z,
                       n_bins_x = 20,
                       n_bins_y = 20,
                       verbose=False,
                       plot=False, cmap='coolwarm'):
    x_grid = np.linspace(np.min(x), np.max(x), n_bins_x)
    y_grid = np.linspace(np.min(y), np.max(y), n_bins_y)
    dx = x_grid[1]-x_grid[0]
    dy = y_grid[1]-y_grid[0]
    x_grid,y_grid = np.meshgrid(x_grid,y_grid)

    histo = np.zeros(x_grid.shape)
    for n0 in range(histo.shape[0]):
        if verbose:
            print(histo.shape[0]-n0,end=' ')
        for n1 in range(histo.shape[1]):
            pts_in_bin = ( np.abs(x-x_grid[n0,n1])<dx ) * ( np.abs(y-y_grid[n0,n1])<dy )
            histo[n0,n1] += np.nanmean(z[pts_in_bin])

    if plot:
        fig,ax = plt.subplots(1,2, figsize=(10,4))
        imgs = []
        imgs.append(ax[0].scatter(x,y,c = z, cmap=cmap))

        extent = [np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)]
        imgs.append(ax[1].matshow(histo, cmap=cmap, extent=extent, aspect='auto',origin='lower', interpolation = 'gaussian'))
        ax[1].xaxis.set_ticks_position('bottom')

        for im, axe in zip(imgs,ax.flatten()):
            divider = make_axes_locatable(axe)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        fig.tight_layout()
    return x_grid, y_grid, histo

def pearson_coef(img1,img2):
    
    x = img1 - np.nanmean(img1)
    y = img2 - np.nanmean(img2)
    numerator = np.sum(x*y)
    denominator = np.sqrt(np.sum(x**2.) * np.sum(y**2.))
    
    return (numerator/denominator)


    
