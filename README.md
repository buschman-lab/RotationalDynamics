# Rotational Dynamics

## Neural Network Model of Rotational Dynamics

The neural network model was designed to simulate observed dynamics in auditory cortex during associative learning. The same analyses applied to the neural data are applied here. For each model run, classifiers are trained and trial data is projected on to the classifiers. Additionally, z-scored firing rate differences and neuron selectivity counts are calculated per model run across the population. 

### Software

Python 3.7

jupyter notebook (for demo)

#### packages
numpy 1.16.2, torch 1.0.1, sklearn 0.21.1 (version 0.19 or later), scipy 1.2.1

### Installing

Download files and place script/notebook(s) into the same folder as helper functions. 

### Running Model

#### Script: network_model_rnn.py

Choose number of model runs - line 43

Choose a directory for model save file - line 45-47

Choose network parameters - defaults match paper

#### Notebook: Model Demo 2020

Ensure all helper .py files are in the same folder as the notebook. 
Run each cell in order. The model parameters are set to run example network models with a range of parameters for the level of association and levels of structure in the rotation. Expected time to run - 10-20 minutes. 

#### Expected output: 
All accuracy of classifier are saved at each time point. Accuracy is measured by AUC or area under the curve. Values above .5 indicate accurate performance. 

A/X Sensory and Memory classifiers should perform well during the first and second time points.

For example: 

```
AX AUC per tp: [0.906425 0.733425]
```

C/C* sensory classifier should perform well during the second time point (tp) only. 

For example:

```
CCprime AUC per tp: [0.489225 0.865125]
```

When Association levels are high, A/X sensory classifier should fail during unexpected trials at time point 2. However, A/X memory classifier should perform well on these trials. 

Full network output example: 

```
New Run - p_index: 5 run: 0
current association level 0.95
current level of structure in rotation 0.49889135254988914


AX AUC per tp: [0.906425 0.733425]
CCprime AUC per tp: [0.489225 0.865125]
A/X Sensory AUC on Unexpected trials: 0.33799999999999997
A/X Memory AUC on Unexpected trials: 0.7418
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

