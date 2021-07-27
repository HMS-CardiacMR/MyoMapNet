MyoMapNet Implementation

This folder contains the code to train and test MyoMapNet

Main implementation: is the main implementation presented in the "Accelerated Cardiac T1 Mapping in Four Heartbeats with Inline MyoMapNet: A Deep Learning Based T1 Estimation Approach" paper

./Data/Demo/demo_Phantom.mat
--Mat file for phantom acquired by MOLLI sequence
./TrainedModels
Four tranined models as presented in our manuscript

Testing.py 
--Python code for generating T1 maps for MOLLI phantom data saved under /Data/Demo/demo_Phantom.mat using one of trained models. 

Training.py 
--Python code for traning model wiht simualted signals

For more details, please do not hesitate to connet me (Rui Guo, RGuo@bidmc.harvard.edu)



Alternative_version: contains an alternative implementation for MyoMapNet for a secondary validation and reproducibility assessment.
