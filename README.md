# Rotational Dynamics

## Neural Network Model of Rotational Dynamics

The neural network model was designed to simulate observed dynamics in auditory cortex during associative learning. The same analyses applied to the neural data are applied here. For each model run, classifiers are trained, z-scored firing rate differences and neuron selectivity counts are calculated.

### Software

Python 3.7

jupyter notebook (for demo)

#### packages
numpy 1.16.2, torch 1.0.1, sklearn 0.21.1, scipy 1.2.1

### Installing

Download files and place script/notebook(s) into the same folder as helper functions. 

### Running Model

#### Script: network_model_rnn.py

Choose number of model runs - line 43

Choose a directory for model save file - line 45-47

Choose network parameters - defaults match paper

#### Notebook: Model Demo 2020

Ensure all helper .py files are in the same folder as the notebook. 
Run each cell in order. The model parameters are set to run example network models with a range of parameters for the level of association and levels of structure in the rotation. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

