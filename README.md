# PMI-CNN
Parallel Multi-Input Mechanism-Based Convolutional Neural Network

## Hyperspectral Image Classification via Parallel Multi-Input Mechanism-Based Convolutional Neural Network
In this paper, a parallel multi-input convolutional neural network (PMI-CNN) is proposed for hyperspectral images classification.

## Overview
HSIs classification aims to classify the surface materials belonging to each pixel in the HSIs. Although spectral information in HSIs provides a solid basis for classification, HSIs’ high-dimensional information is accompanied by some problems, such as data redundancy and dimensional disaster, which increase the computational complexity and reduce classification accuracy. Besides, HSIs usually only contain a small number of labeled samples, and the distribution of samples is unbalanced, which makes HSIs classification more complicated.

## Methodology
We propose a parallel multi-input mechanism-based CNN (PMI-CNN) which makes full use of spectral-spatial information in Hyperspectral Images. The model consists of four parallel convolution branches. Four neighborhood blocks of different sizes are input into four parallel branches of the network to extract spatial features with different levels through independent convolution operations. Finally, each branch’s obtained feature maps are spliced, and the feature maps are used as the classifier’s input.

## How to use the code
###  Environment configuration 
Deep learning framework: Tensorflow2.5.0

### Code configuration
1.Model.py is the main program.
2.HSI_Prepare.py is a data preprocessing program that divides the dataset by percentage.
3.Hisprepare_2.py is also a data preprocessing program, which can divide the dataset according to a certain number of each class.
4.helper.py and process.py are custom tool libraries.

## Dataset
Three publicly available HSI datasets are used to evaluate the model's performance: Indian Pines dataset (IP), Pavia University Dataset (PU) and Salinas Scene Dataset (SA).
