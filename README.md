# MyoMapNet: A Deep Learning Based T1 Estimation Approach in Four Heartbeats

*We implemented a FC that uses pixel-wise T1-weighted signals and corresponding inversion time to estimate T1 values from a limited number of T1-weighted images. After a rigorous training using in-vivo T1 maps of 607 patients, undergoing clinical cardiac MR exams, collected by MOLLI, the performance of MyoMapNet was evaluated using in-vivo data of 61 patients by discarding the additional T1-weighted images from MOLLI. The inline MyoMapNet was then used to collect LL4 T1 and MOLLI in 16 subjects to demonstrate feasibility of inline MyoMapNet.*

## Summary Video

[![Click to watch video](https://img.youtube.com/vi/zrQ7A0B26E0/maxresdefault.jpg)](http://www.youtube.com/watch?v=zrQ7A0B26E0 "A Guide To Deploying Deep Learning Models On Scanner: A Case Study With MyoMapNet")

## Abstract

**Purpose**: To develop and evaluate MyoMapNet, a rapid myocardial T1 mapping approach that uses fully connected neural networks (FC) to estimate pixel-wise T1 from only four T1-weighted images collected after a single inversion pulse (Look-Locker, LL4) in four heartbeats.

**Method**: We implemented an FC that uses pixel-wise T1-weighted signals and corresponding inversion times to estimate T1 value from a reduced number of T1-weighted images. We studied how training the model using native, post-contrast T1 and a combination of both could impact performance of the MyoMapNet. We also explored two choices of the number of T1-weighted images (four and five for native T1, selected to allow training of network using existing data from modified Look-Locker sequence (MOLLI). After a rigorous training using in-vivo MOLLI T1 maps of 607 patients, the performance of MyoMapNet was evaluated using MOLLI T1 data of 61 patients by discarding the additional T1-weighted images. Subsequently, we implemented a prototype MyoMapNet and LL4 on a Siemens 3.0T scanner. LL4 was used to collect T1 mapping of 27 subjects with inline T1 map reconstruction by MyoMapNet and resulting T1 values were compared with MOLLI.

**Results**: MyoMapNet trained using a combination of native and post-contrast T1 demonstrated excellent accuracy between MOLLI and MyoMapNet for native and post-contrast T1. The FC model using four T1 weighted images yields similar performance when compared to five T1 weighted images, suggesting that four T1 weighted images may be sufficient. The inline implementation of MyoMapNet enables successful acquisition and reconstruction of T1 maps via MyoMapNet on the scanner. Myocardial T1 by MOLLI and LL4 with inline MyoMapNet was 1170±55 ms vs. 1183±57 ms (P=0.03), and 645± 26 ms vs 630±30 ms (P=0.60) for native and post-contrast myocardial T1, and 1820±29 ms vs 1854±34 ms (P=0.14), and 508±9 ms vs. 514±15 ms (P=0.02) for native and post-contrast blood T1, respectively.

**Conclusion**: A FC neural network, trained using MOLLI data, can estimate T1 values from only four T1-weighted images. MyoMapNet T1 mapping enables myocardial T1 mapping in four heartbeats with similar accuracy as MOLLI with inline map reconstruction.

**Keywords**: Inversion-recovery cardiac T1 mapping, Machine learning, Cardiac MRI, Myocardial tissue characterization

## Publications

R Guo, H El-Rewaidy, S Assana, et al. ["Accelerated cardiac T1 mapping in four heartbeats with inline MyoMapNet: a deep learning-based T1 estimation approach."](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/guo_etal_jcmr_vol24.pdf) *Journal of Cardiovascular Magnetic Resonance* 24.1 (2022): 1-15.

R Guo, Z Chen, A Amyar, H El‐Rewaidy, S Assana, et al. ["Improving accuracy of myocardial T1 estimation in MyoMapNet."](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/magnetic_resonance_in_med_-_2022_-_guo_-_improving_accuracy_of_myocardia.pdf) *Magnetic Resonance in Medicine* 88.6 (2022): 2573-2582.

A Amyar, R Guo, X Cai, S Assana, et al. ["Impact of deep learning architectures on accelerated cardiac T1 mapping using MyoMapNet."](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/nmr_in_biomedicine_-_2022_-_amyar_-_impact_of_deep_learning_architectures_on_accelerated_cardiac_t1_mapping_using_myomapnet.pdf) *NMR in Biomedicine* 35.11 (2022): e4794.

## Code

The codes of MyoMapNet, including in-line implementation, are openly available on GitHub.

- [Off-Line Implementation](https://github.com/HMS-CardiacMR/MyoMapNet/tree/main/MyoMapNet_Implementation/Main_implementation): Contains the main code used to to train and test MyoMapNet. Four trained moddels are included: 4 Beat Pre-Gad, 4 Beat Post-Gad, 4 Pre&Post-Gad and 5 Beat Pre-Gad.

- [In-Line Implementation](https://github.com/HMS-CardiacMR/MyoMapNet/tree/main/InLine_Implementation): The inline integration is implemented using the Siemens Framework for Image Reconstruction (FIRE) prototype framework. Briefly, the FIRE framework provides an interface for raw data or image data between the Siemens Image Reconstruction Environment (ICE) pipeline and an external environment like Python.

## Data

This publically available dataset contains the data used to accomplish the following: testing our developed and trained MyoMapNet models, comparing our models to other novel neural networks, improving MyoMapNet, and proposing new techniques for myocardial T1 mapping. We shared all testing datasets and corresponding T1 maps of each subject.
 
The testing dataset includes two subsets: Existing MOLLI dataset and Prospective T1 mapping dataset. For the Existing MOLLI dataset, T1 maps were fitted from all MOLLI T1-weighted images by curve-fitting method and reconstructed from the first four or five MOLLI T1-weighted images by MyoMapNet with different inputs. Prospective images were collected using MOLLI and the proposed LL4 sequence. T1 maps were built in same manner as those of Existing MOLLI data. For both MOLLI and LL4, native and post-contrast data are available.
 
Details of this dataset, including how to use and implement this data, are described in the DataDescription document on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5MZYAH).
