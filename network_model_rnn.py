### Network Model
### RNN model with inputs
### options - number of runs and network parameters 
### Alex Libby

#### Python version 3.7
### numpy version - '1.16.2'
import numpy as np
import functools
import operator
import sys
import pickle 

### torch version - '1.0.1'
import torch
import torch.nn as nn

import random

# sklearn.__version__  - '0.21.1'
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

# scipy.__version__ - '1.2.1'
from scipy.special import ndtri

import datetime
from network_helper_functions import *

#############################################
#############################################
####### Run Model 
####### updated - 8/30/2019 
####### updated - 4/8/2020 

### save model output 
print('starting script')
sys.stdout.flush()
array_index = int(sys.argv[1])
print('current run: '+str(array_index))

num_runs = 1 
plot_check = False
folder_name = 'model_run_all_linkrot' #'model_run_noise'
#### save model results
save_dir = '/jukebox/buschman/Users/Alex/python_spock_model/save_files/'+folder_name+'/'
#####################################################################
network_parm = {}
network_parm['N'] = 150
network_parm['num_tp_on'] = 1#5 ### num tp per stim that stim is on
network_parm['num_tp_stim'] = 1#10 ### num tp per stim
network_parm['num_trials'] = 1000 ### number of trials 
network_parm['permute_trials'] = False ### doesn't matter unless you have learning
network_parm['model_type'] = 'rec' 

### activations in the input layer (current size = 4 units) 
input_dict = {'stim0':{'A':np.tile(np.array([1,0,0,0]).reshape(-1,1),network_parm['num_tp_on']),\
              'X':np.tile(np.array([0,1,0,0]).reshape(-1,1),network_parm['num_tp_on'])},\
              'stim1':{'C':np.tile(np.array([0,0,1,0]).reshape(-1,1),network_parm['num_tp_on']),\
             'C*':np.tile(np.array([0,0,0,1]).reshape(-1,1),network_parm['num_tp_on'])}}

network_parm['num_stim'] = len(input_dict.keys())
network_parm['input_size'] = len(input_dict['stim0']['A'])
network_parm['num_tp'] = network_parm['num_stim']*network_parm['num_tp_stim']

### Classifier information
clf_fit = ['AX','CCprime']
network_parm['tp_win'] = 1 ### train at each tp but with a window - for classifier training
network_parm['tp_win_zscore'] = 1 ### for z-score calculation 
#### taken from - clf train oct 2018
loss_function = 'hinge'
best_ratio = .65 ### l1 ratio - in the elastic net
best_alpha = .01 ### regularization amount 
best_n_iter = 1000
best_learn_rate = .00001
opt_params = {'loss':loss_function,'penalty': 'elasticnet', 'l1_ratio': best_ratio, 'alpha':best_alpha,
                                      'max_iter':best_n_iter,'learning_rate':'constant','eta0':best_learn_rate}
clf_class = SGDClassifier
test_percent_y = .1 # how much you test on
#####################################################################
#### For network_parm - indicate the range of parameters 
#### noise level added at each time step - see methods 
#### transfer_function - activation function - see methods
#### link_percents - association between A/X and C/C* - indicates percentage of C/C* neurons linked
##### - note code will round to keep C/C* and A/X selectivity constant 

##### link_std - variance within a representation on-diagonal variance of covariance matrix - see methods 
##### rand_std_all - off-diagonal variance - level of structure in the rotation. 
        ### 0 is random. when rand_std_all == link_std - the rotation is highly structured
        ### structured rotation has stable and switching neurons
        
##### hold_sel_count -- control the number of neurons selective to an input and memory
change_p_dict = {'noise_level':[2],'tau':[1],'transfer_func':['Relu'],\
                'link_percents':np.linspace(0,.95,10),'link_std':[.451],\
                 'rand_std_all':np.linspace(0,.45,50),\
                 'hold_sel_count':[50]} 
                 
#####################################################################
#####################################################################
################# START NETWORK RUNS ################################
#network_parm['network_min'] = -1
all_parm_val,num_p = network_parm_grid(change_p_dict)
print('num p: '+str(num_p))

##### PRE ALLOCATION 
all_tp_auc = {}
all_tp_coeff = {}
all_tp_intercept = {}
all_tp_zscore = {}
for current_clf_name in clf_fit:
    all_tp_auc[current_clf_name] = np.zeros((num_p,num_runs,network_parm['num_tp']-network_parm['tp_win']+1))
    all_tp_coeff[current_clf_name] = np.zeros((num_p,num_runs,\
                                    network_parm['num_tp']-network_parm['tp_win']+1,network_parm['N']))
    all_tp_intercept[current_clf_name] = np.zeros((num_p,num_runs,\
                                                  network_parm['num_tp']-network_parm['tp_win']+1))   
    all_tp_zscore[current_clf_name] = np.zeros((num_p,num_runs,\
                                    network_parm['num_tp']-network_parm['tp_win']+1,network_parm['N']))
    
W_all = np.zeros((network_parm['N'],network_parm['N'],num_p,num_runs)) ### for saving later
IH_all = np.zeros((network_parm['input_size'],network_parm['N'],num_p,num_runs)) ### for saving later
AX_change_all = np.zeros((4,network_parm['N'],num_p,num_runs)) ### for saving later
counts_all = np.zeros((num_p,num_runs,9))

#### pre-allocation for y-score - added 5/13/2020 - AL
y_score_all = {}
for current_clf_name in clf_fit:
    y_score_all[current_clf_name] = np.zeros((num_p,num_runs,network_parm['num_tp']-network_parm['tp_win']+1,\
                                             network_parm['num_tp'],network_parm['num_trials']))
                                             
#### preallocate angle space - 4/6/2020 - AL
angle = {}
comp_compare = [['AX_sen','CCp'],['AX_sen','AX_mem'],['AX_mem','CCp']]
for comp in comp_compare:
    angle['_'.join(comp)] = np.zeros((num_p,num_runs))
    
#### pre allocation for cross auc (8/30/2019 - AL)###############
clf_comp = ['AX','CCprime']
cross_auc = {}
#### clf train
for clf_type in clf_comp:
    #### clf train period
    for tp_train in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):
        clf_train_name = clf_type+str(tp_train)
        cross_auc[clf_train_name] = {}
        ##################################
        #### test response comparison type
        for current_cond_type in clf_comp:
            #### test response time period (tp)    
            for tp_test in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):

                test_name = current_cond_type+str(tp_test)
                cross_auc[clf_train_name][test_name] = np.zeros((num_p,num_runs))
                
################################################
#### POSTDICTION pre-allocation (8/30/2019 - AL)
tp_test = network_parm['num_tp_stim'] ### only look at the second time period 
eu_auc = {}
#### clf train
for clf_type in clf_comp:

    #### clf train period
    for tp_train in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):

        clf_train_name = clf_type+str(tp_train)
        eu_auc[clf_train_name] = {}

        ##################################
        #### test response comparison type
        for current_cond_type in clf_comp:

            for exp_comp in ['exp','unexp']:

                test_name = current_cond_type+str(tp_test)+exp_comp
                eu_auc[clf_train_name][test_name] = np.zeros((num_p,num_runs))
                
#####################################################################
#####################################################################
### run / parameter index loop

for run in np.arange(num_runs):
    for p_index in np.arange(num_p):
        ### set network parameters 
        for change_name in all_parm_val.keys():
            network_parm[change_name] = all_parm_val[change_name][p_index]

        #### set C/C* selectivity to the selectivity count
        network_parm['CCp_sel'] = network_parm['hold_sel_count']

        #StSw_ratio = 1
        num_trials_total = 1000
        count_d = int(np.round(network_parm['hold_sel_count']/2))
        none_count = int(network_parm['N'] - 2*count_d)
        cell_type_counts = {'single_0': 0,'single_1': 0,\
                                        'switch': count_d,'stable': count_d,\
                                        'none': none_count,'rand_sel': 0,'random_uni': 0,'random_norm': 0.0}

        #############################    
        print('\nNew Run - p_index: '+str(p_index)+' run: '+str(run))
        sys.stdout.flush()
        #print('\nNEW RUN: num_cells '+str(network_parm['N']))

        ### 2) INPUT MATRIX (trial types and input layer)
        input_matrix,trial_types = define_network_inputs(network_parm,input_dict)
        num_trial_types,input_feature,num_tp = input_matrix.shape
        #print('num_trial types: '+str(num_trial_types)+', input feature: '+str(input_feature)+', num tp: '+str(num_tp))

        ### plotting check --
        if np.all([plot_check==True,run==0]):
            for trial_index in np.arange(len(trial_types)):
                fig = plt.figure(figsize=(4,4))
                ah = plt.gca()
                plt.title(trial_types[trial_index])
                mesh_data(ah,input_matrix[trial_index,:,:],network_parm['num_tp_on'],1)

        ### 3) CONDITION TRIAL LIST:
        condition_list,num_trial_total = return_cond_list(network_parm['num_trials'],num_trial_types,\
                                                          random_on=network_parm['permute_trials'])
        
        #print('condition set '+str(np.unique(condition_list,return_counts=True))) 

        ####################################################
        ##### RNN CONNECTIONS - W - 3/16/2020 - Al 
        #### generate AX change weight matrix - 3/16/2020 - AL
        AX_change = weight_array_generate(cell_type_counts,rand_std=network_parm['rand_std_all'],\
                                          link_std=network_parm['link_std'])
        AX_change_all[:,:,p_index,run] = AX_change
        #AX_change[2:4,:].T = np.dot(W,AX_change[0:2,:].T) 

        ### set responses to A/X during the sensory period
        AX_sen = AX_change[0:2,:]

        ### set responses to A/X during the memory period
        AX_mem = AX_change[2:4,:]

        ### solve for W:
        W  = np.dot(AX_mem.T,np.linalg.pinv(AX_sen.T))
        W_all[:,:,p_index,run] = W #### save the W matrix - added 4/2/2020 - AL

        ### input 
        IH = associate_IH(AX_sen,network_parm)
        IH_all[:,:,p_index,run] = IH
        
        ##################################################
        #### DEFINE NETWORK 
        net = TL(network_parm,IH,W)
        ##net.IH.shape
        response_all = np.zeros((network_parm['num_trials'],network_parm['N'],num_tp))

        ################ RUN NETWORK FORWARD ############################
        #print('running network')
        for trial in np.arange(network_parm['num_trials']):

            cc = int(condition_list[trial]) #### get current condition

            Vt = np.zeros(network_parm['N'])

            for tp in np.arange(num_tp):

                current_input = input_matrix[cc,:,tp]
                Vt = net.forward(Vt,network_parm,current_input)
                response_all[trial,:,tp] = Vt
        ##################################################################
        
        
        ##################################################################
        ####### CLASSIFICATION (7/31/2019 -AL)
        ##################################################################
        ### Condition and Classifier paramaters 
        ### checked - 4/2/2020 - AL
        cond_comp_dict = {'AC_seq':{'AX':{0:[0,1],1:[2,3]},'CCprime':{0:[0,2],1:[1,3]}}} 
        exp_dict = {'exp':{'AX':{0: [0], 1: [3]},'CCprime':{0: [0], 1: [3]}},\
                    'unexp':{'AX':{0: [1], 1: [2]},'CCprime':{0: [2], 1: [1]}}} 
        tp_clf_start = {'CCprime':network_parm['num_tp_stim'],'A/X mem':network_parm['num_tp_stim'],'A/X':0}

        #### for the angle calculations - 4/6/2020 - AL
        dict_axes = {'AX_sen':{'name':'AX','tp':0*network_parm['num_tp_stim']},\
                     'AX_mem':{'name':'AX','tp':1*network_parm['num_tp_stim']},\
                     'CCp':{'name':'CCprime','tp':1*network_parm['num_tp_stim']}}
        ##################################################################

        cond_train_labels = np.unique(condition_list)

        ###### Test Train Split
        test_trial_all,train_trial_all_use = split_train_test(condition_list,\
                                                test_percent_y,cond_train_labels)


        #### loop through classifiers - 4/2/2020 - AL
        all_clf = {}
        for current_clf_name in ['AX','CCprime']:

            condition_list_bin = return_bin_condition_list(condition_list,current_clf_name,\
                                                           cond_comp_dict,paradigm_type='AC_seq')

            clf,tp_auc = clf_train_test(test_trial_all,\
                                train_trial_all_use,clf_class,opt_params,condition_list_bin,\
                                        response_all,network_parm['tp_win'])

            
            ##### y-score add in - 5/13/2020 - AL
            for tp_train in list(clf.keys()):
                for tp_test in np.arange(num_tp):
                    y_score_all[current_clf_name][p_index,run,tp_train,tp_test,:] = \
                    clf[tp_train].decision_function(response_all[:,:,tp_test])
            #############################################################################
            
            #if run==1:
            print(current_clf_name+' AUC per tp: '+str(tp_auc))
            sys.stdout.flush()

            ### save clf locally... 4/6/2020 - AL
            all_clf[current_clf_name] = clf
            
            #### save auc data 
            #all_tp_auc[current_clf_name][p_index,run,:] = tp_auc

            ### save coeff - 4/3/2020 - Al
            for tp in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):
                all_tp_coeff[current_clf_name][p_index,run,tp,:] = clf[tp].coef_[0]
                all_tp_intercept[current_clf_name][p_index,run,tp] = clf[tp].intercept_[0]

            ##################################################################    
            #### Z-SCORE calculation - updated 4/3/2020 - AL - shape - tp x N
            zscore_diff = zscore_activity(response_all,condition_list,condition_list_bin,\
                                                network_parm,num_shuffles = 1000)

            #### save z-score
            all_tp_zscore[current_clf_name][p_index,run,:,:] = zscore_diff
            ################################################################## 

            ### get z-score blocks - added 4/3/2020 - Al
            zscore_diff_periods = return_zscore_blocks(zscore_diff,network_parm)

            ### make the AX table
            if current_clf_name=='AX':

                counts,counts_keys = create_cell_counts_table(zscore_diff_periods)
                counts_all[p_index,run,:] = counts
                
         
        #### UPDATE OTHER METRICS - 4/6/2020 - AL
        angle = cal_angle_axes(angle,all_clf,comp_compare,dict_axes,p_index,run)
        
        cross_auc = cross_auc_clf(cross_auc,clf_comp,num_tp,all_clf,condition_list,\
                                  cond_comp_dict,test_trial_all,p_index,run,network_parm,response_all)
        
        eu_auc = postdiction_test(eu_auc,clf_comp,all_clf,exp_comp,exp_dict,condition_list,\
                                  test_trial_all,response_all,tp_test,\
                                  network_parm,cond_comp_dict,p_index,run)
        
        print('AX0 --> AX1unexp: '+str(eu_auc['AX0']['AX1unexp'][p_index,run]))
        print('AX1 --> AX1unexp: '+str(eu_auc['AX1']['AX1unexp'][p_index,run]))
        sys.stdout.flush()


###### SAVE MODEL #############################################################################
fn = 'model_save_2020_run_'+str(array_index)+'.pkl'
f = open(save_dir+fn,'wb')

#### other variables to save: 
#### all_tp_drop_auc,drop_eu_auc,all_tp_auc
pickle.dump([num_runs,network_parm,input_dict,clf_fit,change_p_dict,all_parm_val,num_p,\
all_tp_coeff,all_tp_intercept,all_tp_zscore,W_all,IH_all,counts_all,\
angle,clf_comp,cross_auc,tp_test,eu_auc,AX_change_all,y_score_all,condition_list],f,protocol=4)

f.close()
print('saved model results')
sys.stdout.flush()

