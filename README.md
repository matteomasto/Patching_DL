# Patching_DL

This repository contains the codes for the implementation using Tensorflow v.2.2.0 of a deep learning model for the inpainting of 3D Bragg X-raydiffraction patterns 
I developped the package during my PhD.

The package allows for the handling of the three main stages of a BCDI data processing workflow:

    the proprocessing (data centering and cropping)
    the phase retrieval using PyNX.
    the post processing (orthogonalization, phase manipulation, strain computation etc.)

It is assumed that the phase retrieval is carried out by the PyNX package (see http://ftp.esrf.fr/pub/scisoft/PyNX/doc/). The BcdiPipeline class runs the three stages and can manage connection to different machines if required (GPU for phase retrieval).

Pre- and post-processing do not require GPUs and can be run using the present package (cdiutils backend) or the bcdi package (see https://github.com/carnisj/bcdi) (bcdi backend).

The package also provide utility fonctions to analyze processed data and plot them for potential publications.
