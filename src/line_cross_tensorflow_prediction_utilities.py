import numpy as np
import pylab as plt

#####################################################################################################################################
#############################                    Line prediction                    #################################################
#####################################################################################################################################

import math
def prediction_line_gap(data_masked_original, mask_original, gap_params,
                        model,
                        skip_pixels=0):
    
    axis_perpendicular = gap_params['axis_perpendicular']
    gap_size = gap_params['gap_size']
    mask_position = math.floor(gap_params['mask_position']) 

    # I use this floor since it's fine for the 6 pixels gap. I don't know how fine it is for other gap size unfortunatly.
    
    # Force the mask be perpendicular to axis 1
    if axis_perpendicular != 1:
        data_masked = np.swapaxes(data_masked_original, axis_perpendicular,1)
        mask = np.swapaxes(mask_original, axis_perpendicular,1)
    else:
        data_masked = np.copy(data_masked_original)
        mask = np.copy(mask_original)
        
#     plot_3D_projections(data_masked)
    
    prediction  = np.copy(data_masked)
    average_denominator = np.zeros(data_masked.shape)
    
    size_pred = model.output_shape[1] # size of the model prediction
    
    # Axes to move the prediction window (parallels to the gap plane)
    axis_list = np.arange(3)
#     axis_list = np.delete(np.arange(3), axis_perpendicular)
    axis_list = np.delete(np.arange(3), 1)
    
    m = 0
    final_m = 0 # Something to avoid empty prediction near the edges
    while m+size_pred <= data_masked.shape[axis_list[0]] and final_m<=1: 

        n = 0
        final_n = 0 # Something to avoid empty prediction near the edges
        while n+size_pred <= data_masked.shape[axis_list[1]] and final_n<=1:

            X = np.zeros((1, size_pred, size_pred, size_pred, 1))
            
            s = [slice(None) for n in range(3)]
            s[axis_list[0]] = slice(m, m+size_pred)
            s[axis_list[1]] = slice(n, n+size_pred)
#             s[axis_perpendicular] = slice(mask_position-size_pred//2+1, mask_position+size_pred//2+1)
            s[1] = slice(mask_position-size_pred//2+1, mask_position+size_pred//2+1)
            s = tuple(s)
            
            X[0,...,0] = data_masked[s]
            small_mask = mask[s]

#             maxi = X[0,...,0].max()
#             mini = X[0,...,0].min()
            maxi = X[0,...,0][small_mask==0].max()
            mini = X[0,...,0][small_mask==0].min()
            X[0,...,0] = np.interp(X[0,...,0], (mini,maxi),(0,1))
            
#             if n>40 and m>40:
#                 return X[0,...,0]

            y_pred = model.predict(X,verbose = 0)[0,...,0]
            ####
            y_pred = y_pred*small_mask + X[0,...,0]*(1.-small_mask)
            ####
            y_pred = np.interp(y_pred, (0,1),(mini,maxi))

            prediction[s] += np.copy(y_pred*small_mask)
            average_denominator[s] += np.copy(small_mask)

            if n+1+skip_pixels+size_pred<= data_masked.shape[axis_list[1]]:
                n += 1+skip_pixels
            else:
                n = data_masked.shape[axis_list[1]] - size_pred # To avoid zeros in the prediction
                final_n += 1

        if m+1+skip_pixels+size_pred<= data_masked.shape[axis_list[0]]:
            m += 1+skip_pixels
        else:
            m = data_masked.shape[axis_list[0]] - size_pred # To avoid zeros in the prediction
            final_m += 1
        print(m, end = ' ')

    predicted_pixels = (average_denominator!=0)
    average_denominator[average_denominator==0] = 1 
    # I'll divide the whole prediction array so I need to divide by 1 where I'm not changing anything
    prediction = np.divide(prediction, average_denominator)
    
    if axis_perpendicular != 1:
        prediction = np.swapaxes(prediction, 1, axis_perpendicular)
        predicted_pixels = np.swapaxes(predicted_pixels, 1, axis_perpendicular)
    
    return prediction, predicted_pixels

#####################################################################################################################################
#############################                    cross prediction                    ################################################
#####################################################################################################################################

def prediction_cross_gap_axisparallel2(data_masked, mask, gap_params, 
                         model,
                         skip_pixels=0,
                        clean_cross_region=True):

    axis_list = np.arange(3)
    axis_list = np.delete(np.arange(3), gap_params['axis_parallel'])
    
    gap_prediction = np.zeros(data_masked.shape)
    average_denominator = np.zeros(data_masked.shape)
    for axis in axis_list:
        gap_params_temp = {'axis_perpendicular' : axis,
                           'gap_size' : gap_params['gap_size'],
                           'mask_position' : gap_params['mask_position'][axis]}
        prediction_temp, predicted_pixels_temp = prediction_line_gap(data_masked, mask, gap_params_temp,
                                                                     model,
                                                                     skip_pixels=skip_pixels)
        
        if clean_cross_region :
            mask_cleaning_cross = np.zeros(data_masked.shape)
            s = [slice(None) for n in range(3)]
            s[axis] = slice(min(gap_params['pixel_masked'][axis]), max(gap_params['pixel_masked'][axis])+1)
            s = tuple(s)
            mask_cleaning_cross[s] += 1
            gap_prediction += np.copy(prediction_temp*mask_cleaning_cross)
            average_denominator += np.copy(mask_cleaning_cross)
        else:
            gap_prediction += np.copy(prediction_temp*mask)
            average_denominator += np.copy(predicted_pixels_temp)
        
    average_denominator[average_denominator==0] = 1    
    gap_prediction = np.divide(gap_prediction,average_denominator)
    prediction = np.copy(data_masked) + gap_prediction
    return prediction

def prediction_cross_gap(data_masked, mask, gap_params, 
                         model,
                         skip_pixels=0,
                        clean_cross_region=True):
    if gap_params['axis_parallel'] == 2:
        prediction = prediction_cross_gap_axisparallel2(data_masked, mask, gap_params, 
                         model,
                         skip_pixels=skip_pixels, clean_cross_region=clean_cross_region)
    else:
        data_masked_temp = np.swapaxes(data_masked, gap_params['axis_parallel'],2)
        mask_temp = np.swapaxes(mask, gap_params['axis_parallel'],2)
        gap_params_temp = gap_params.copy()
        gap_params_temp['axis_parallel'] = 2
        gap_params_temp['mask_position'][[gap_params['axis_parallel'], 2]] = gap_params_temp['mask_position'][[2, gap_params['axis_parallel']]]
        gap_params_temp['pixel_masked'][[gap_params['axis_parallel'], 2]] = gap_params_temp['pixel_masked'][[2, gap_params['axis_parallel']]]
        prediction = prediction_cross_gap_axisparallel2(data_masked_temp, mask_temp, gap_params_temp, 
                         model,
                         skip_pixels=skip_pixels, clean_cross_region=clean_cross_region)
        prediction = np.swapaxes(prediction, 2, gap_params['axis_parallel'])
    return prediction

def gap_prediction(data_masked, mask, gap_params, model,
                   skip_pixels=0):
    if gap_params['gap_shape'] == 'line':
        prediction, predicted_pixels= prediction_line_gap(data_masked, mask, gap_params, model,
                                                          skip_pixels=skip_pixels)
    if gap_params['gap_shape'] == 'cross':
        prediction = prediction_cross_gap(data_masked, mask, gap_params, model,
                                          skip_pixels=skip_pixels,
                                          clean_cross_region=True)
    return prediction