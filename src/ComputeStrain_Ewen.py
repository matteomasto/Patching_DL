from skimage.restoration import unwrap_phase as unwrap_phase_skimage
def get_cropped_module_phase(obj,
                             threshold_module = None, support = None,
                             crop=True, apply_fftshift=False, unwrap=True):
    
    if apply_fftshift:
        obj = fftshift(obj)
        
    shape = obj.shape
    if crop:
        obj = crop_array_half_size(obj)
        if support is not None:
            support = crop_array_half_size(support)
        
    module = np.abs(obj)
    
    if support is None:
        if threshold_module is None:
            if obj.ndim ==3:
                threshold_module=.01 # Seems that 3D data need a smaller threshold
            if obj.ndim == 2:
                threshold_module = .3
        support = (module >= np.nanmax(module)*threshold_module)        
    
    phase = np.angle(obj)
    
    if unwrap:
        mask_unwrap = (1-support)
        
        if np.any(np.isnan(obj)): # Need to take nan's into account
            print('nan\'s are stil la problem, this freezes the unwrapping')
#             mask_nan = np.isnan(obj)
#             mask_unwrap = mask_unwrap + mask_nan
#             mask_unwrap[mask_unwrap != 0] = 1
#  Fail to take nan's into account...
            
        phase = np.ma.masked_array(phase, mask=mask_unwrap)
        phase = unwrap_phase_skimage(
                phase,
                wrap_around=False,
                seed=1
            ).data

#     if unwrap:
#         phase = unwrap_phase(obj)
#     else:
#         phase = np.angle(obj)
 
    phase[support==0] = np.nan 

    
#     # Badly written Not a good thing actually is the center is a nan
#     if phase.ndim==2:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2]
#     if phase.ndim==3:
#         phase -= phase[phase.shape[0]//2,phase.shape[1]//2,phase.shape[2]//2]
        
    return module, phase


def EB_custom_gradient(array, 
                       voxel_sizes=None):
    '''
    Should work for any array dimensions
    '''
    
    grad = np.zeros((array.ndim,)+array.shape)
    for n in range(array.ndim):
        slice1 = [slice(None) for n in range(array.ndim)]
        slice1[n] = slice(1,None)

        slice2 = [slice(None) for n in range(array.ndim)]
        slice2[n] = slice(None,-1)

        grad_n = array[tuple(slice1)] - array[tuple(slice2)]
        grad_n = np.nanmean([grad_n[tuple(slice1)], grad_n[tuple(slice2)]], axis=0)

#         padding = np.zeros((array.ndim, array.ndim)).astype('int')
        padding = np.zeros((array.ndim, 2)).astype('int')
        padding[n] += 1
        padding = tuple(map(tuple, padding))
        grad_n = np.pad(grad_n, padding, 'constant', constant_values=(np.nan))

        if voxel_sizes is not None:
            grad_n = grad_n/voxel_sizes[n]
            
        grad[n] += grad_n

    return grad