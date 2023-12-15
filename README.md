# Patching_DL

This repository contains the codes and the instructions for the implementation using Tensorflow v.2.2.0 of a deep learning model for the 3D image inpainting. It has been specifically designed for 3D Bragg X-ray diffraction patterns affected by detectors gaps. 

The structure of the repository is the following:  

**- data:** BCDI files  
**- models:** saved weights of the models for the inpainting of gaps with size of 3, 6, 9, 12 pixels  
**- notebooks:** Jupyter notebooks for the handling of the model, the plotting of figures and assessment of model performance  
**- src:** Jupyter notebooks for the creation/training of the models and for the application to entire BCDI patterns.


I developped the codes during my PhD at the University Grenoble - Alpes and at the ID01 beamline of the European Synchrotron Radiation Facility (ESRF-EBS). The PhD is also part of the ENGAGE programme, thus partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement number 101034267.


![DynamicInpainting](https://github.com/matteomasto/Patching_DL/assets/137916908/488002d2-f21f-45c1-8f8c-27de0e93057f)
