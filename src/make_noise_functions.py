import numpy as np
from numpy.fft import fftn, ifftn, ifftshift, fftshift

def force_poisson_statistic(obj, 
                            scale_poisson=None,
                            seed = False,
                            plot=False):
    if scale_poisson is None:
#         scale_poisson_power = np.random.uniform(5,6.3)  # CHOSEN BY EB. Don't hesitate to change
        scale_poisson_power = 6
        scale_poisson = 10**scale_poisson_power
        
    Fexp = ifftshift(fftn(fftshift(obj))) # diffracted complex amplitude
    I = np.abs(Fexp)**2. # diffracted intensity (what we mesure experimentally)
    
    if seed:
        np.random.seed(3)
        
    I_poisson = np.random.poisson(lam = I * scale_poisson / np.max(I)).astype('float64') # apply poisson statistic
    Fexp_poisson = np.sqrt(I_poisson) * np.exp(1.0j * np.angle(Fexp)) # apply this poisson on the diffracted module
    obj_poisson = ifftshift(ifftn(fftshift(Fexp_poisson))) # get corresponding object
    
    mask_zeros = (I_poisson==0.)
    
    if plot:
        plot_diffraction_3d(I, fig_title='diffracted intensity')
        plot_2D_slices_middle(obj, threshold_module=.3, fig_title='corresponding object')
        plot_diffraction_3d(I_poisson, fig_title='diffracted intensity with poisson statistic')
        plot_2D_slices_middle(obj_poisson, threshold_module=.3, fig_title='corresponding object after poisson statistic')
        
    return obj_poisson, mask_zeros

from numpy.fft import fftn, ifftshift, fftshift
def add_random_noise(obj, seed = False,
                     replace_phase_by_random = True,
                     gaussian_blur = True,
                     poisson_noise = True,
                     plot=False):
    
    if plot:
        Fexp = ifftshift(fftn(fftshift(obj))) # diffracted complex amplitude
        I = np.abs(Fexp)**2. # diffracted intensity (what we mesure experimentally)
        plot_diffraction_3d(I, fig_title='diffracted intensity')
        plot_2D_slices_middle(obj, fig_title='real space object', threshold_module=.3)
    
    # Smooth real space module
    obj = smooth_object(obj, sigma_gaussian=None)
    obj = remove_real_space_module_out_support(obj, threshold_module=.1)
    
    # add random noise to the real space module
    module = np.abs(obj)
    module_noise = add_random_noise_module(module, module_factor = None, correlation_length=None)
    obj = module_noise * np.exp(1.0j*np.angle(obj))
    
    if replace_phase_by_random:
        obj = replace_phase_by_random_phase(obj, phase_range=None, correlation_length=None)
        
    if gaussian_blur:
        obj = Gaussian_blur(obj, sigma= None)
        
    if poisson_noise:
        obj, mask_zeros = force_poisson_statistic(obj, seed = seed, scale_poisson=None)
        
    # Calculate corresponding diffracted amplitude
    Fexp = ifftshift(fftn(fftshift(obj))) 
    if poisson_noise:
        Fexp[mask_zeros] = 0.
        
    if plot:
        I = np.abs(Fexp)**2.  
        plot_diffraction_3d(I, fig_title='diffracted intensity after random noises')
        plot_2D_slices_middle(obj, fig_title='real space object after random noises', threshold_module=.3)

    return obj, Fexp

from scipy.ndimage import gaussian_filter
def smooth_object(obj, 
                  sigma_gaussian=None,
                  plot=False):
    
    if sigma_gaussian is None:
        sigma_gaussian = np.random.uniform(.45, .75) # CHOSEN BY EB. Don't hesitate to change
        
    module = np.abs(obj)
    module_smooth = gaussian_filter(module, sigma=sigma_gaussian)
    obj_smooth = module_smooth * np.exp(1.0j * np.angle(obj))
    
    if plot:
        plot_2D_slices_middle_only_module(obj, fig_title='object module')
        plot_2D_slices_middle_only_module(obj_smooth, fig_title='object module after smoothing')
    return obj_smooth

def remove_real_space_module_out_support(obj,
                                         threshold_module=.1,  # CHOSEN BY EB. Don't hesitate to change
                                         plot=False):
    module = np.abs(obj)
    support = np.array(module > np.max(module)*threshold_module, dtype='int')
    module[support==0] = 0.
    
    obj_clean = module * np.exp(1.0j*np.angle(obj))
    
    if plot:
        plot_2D_slices_middle_only_module(support, fig_title='support using a threshold on the module')
        plot_2D_slices_middle_only_module(obj_clean, fig_title='module after removing everything ouside the support')
    return obj_clean

from scipy.signal import fftconvolve
def random_noise(size, 
                 correlation_length = None, 
                 noise_scale = 1.,
                 plot=False):
    
    if correlation_length is None:
        correlation_length = np.random.uniform(.01,.1)  # CHOSEN BY EB. Don't hesitate to change
    
    f = np.random.normal(size = (size, size, size))
    x,y,z = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size), np.linspace(-1,1,size))
    kernel = np.exp(-x**2./(2.*correlation_length**2.) -y**2./(2.*correlation_length**2.) - z**2./(2.*correlation_length**2.))
    
    noise = fftconvolve(kernel, f, mode='same')
    
    noise = noise * noise_scale/np.max(noise)
    
    if plot:
        plot_2D_slices_middle_one_array3D(noise)
    return noise

def add_random_noise_module(module,
                            module_factor = None,
                            correlation_length=None,
                            plot=False):
    
    if module_factor is None:
        module_factor = np.random.uniform(.5,3.)  # CHOSEN BY EB. Don't hesitate to change
    
    module_range = module_factor*np.max(module)


    module_var = random_noise(module.shape[0], 
                             correlation_length = correlation_length)

    module_var = module_range*(module_var-np.min(module_var))/(np.max(module_var)-np.min(module_var))
    
    module_var = module_var * module/np.max(module) # Make sure to add only where the module is not 0
    
    if plot:
        plot_2D_slices_middle_only_module(module_var, fig_title='noise added to module')
        plot_2D_slices_middle_only_module(module+module_var, fig_title='module + noise')
    return module+module_var

from numpy import pi
def replace_phase_by_random_phase(obj,
                                  phase_range=None,
                                  correlation_length=None, 
                                  plot=False):

    if phase_range is None:
        phase_range = np.random.uniform(pi/3., 3.*pi)  # CHOSEN BY EB. Don't hesitate to change
        # No idea what range I should use here. 
        # Maybe check phase range on experimental data to make sure I'm going far enough.
        
    phase = random_noise(obj.shape[0], 
                             correlation_length = correlation_length)

    phase = phase_range*(phase-np.min(phase))/(np.max(phase)-np.min(phase))-phase_range/2.
    
    obj_new_phase = np.abs(obj) * np.exp(1.0j*phase)

    if plot:
        plot_2D_slices_middle(obj_new_phase, fig_title='object with replaced phase')
    return obj_new_phase 

import scipy as scipy
from scipy import ndimage
from numpy.fft import fftn, ifftn, ifftshift, fftshift

def Gaussian_blur(obj, 
                 sigma=None, 
                 plot=False):
    if sigma is None:
        sigma = np.random.uniform(0.5,1.1)  # CHOSEN BY MM. Don't hesitate to change
        
        
    Fexp = ifftshift(fftn(fftshift(obj))) # diffracted complex amplitude
    I = np.abs(Fexp)**2. # diffracted intensity (what we mesure experimentally)
    I_blur = ndimage.gaussian_filter(I,sigma).astype('float64') # apply poisson statistic
    
    Fexp_blur = np.sqrt(I_blur) * np.exp(1.0j * np.angle(Fexp)) # apply this poisson on the diffracted module
    obj_blur = ifftshift(ifftn(fftshift(Fexp_blur))) # get corresponding object
    
    if plot:
        plot_diffraction_3d(I, fig_title='diffracted intensity')
        plot_2D_slices_middle(obj, threshold_module=.3, fig_title='corresponding object')
        plot_diffraction_3d(I_blur, fig_title='diffracted intensity with Gaussian blur')
        plot_2D_slices_middle(obj_blur, threshold_module=.3, fig_title='corresponding object after Gaussian blur')
        
    return obj_blur

