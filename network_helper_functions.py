#### Network Model of Rotation
#### place in folder with network_model_rnn.py
#### 7/22/2019 - AL
### 3/2/2020 - Al

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
#############################################################
#############################################################

#### returns the index of true values (added 8/29/2018 - AL)
def return_true_index(vector):
    index = [i for i, x in enumerate(vector) if x]
    return(index)
    

### Dynamically set network parameter grid space - 12/30/2019
#### go through and replace each variable range with a tiled range, based on other values
def network_parm_grid(change_p_dict):

    #### get the full parameter space (12/12/2019 - AL)
    num_p = functools.reduce(operator.mul,[len(v) for v in change_p_dict.values()])

    ### get variables to loop through (12/30/2019 - Al)
    loop_var = [i for i in change_p_dict.keys() if len(change_p_dict[i])>1]
    other_var = [i for i in change_p_dict.keys() if len(change_p_dict[i])==1]

    all_parm_val = change_p_dict.copy()

    if len(loop_var)>0:
        var = loop_var[0]
        current_len = len(change_p_dict[var])
        all_parm_val[var] = np.tile(all_parm_val[var],int(num_p/current_len))
        #print(all_parm_val[var])

        for var in loop_var[1:]:
            all_parm_val[var] = np.tile(np.expand_dims(np.array(all_parm_val[var]),1),current_len).flatten()
            current_len = len(all_parm_val[var])
            all_parm_val[var] = np.tile(all_parm_val[var],int(num_p/current_len))
            #print(all_parm_val[var])

        ### tile other variables
        for var in other_var:
            all_parm_val[var] = np.tile(all_parm_val[var],num_p)
            #print(all_parm_val[var])
        
    return(all_parm_val,num_p)

def logistic_sigmoid(x,gain,center=0,L=1):
    out = L/(1+np.exp(-gain*(x-center)))
    return(out)

def transfer_func(x,func_type='Relu',center=100,gain=.1,L=20):
    if func_type=='sigmoid':
        out = logistic_sigmoid(x,gain,center,L)
        out[x <= 0] = 0
        
    elif func_type=='Relu':
        center = 0 ### min firing rate 
        out = center*np.ones(x.shape)
        out[x >= center] = x[x >= center]
        
    elif func_type=='log':
        out = np.zeros(len(x))
        out[x>0] = np.log(3*x[x>0]+1)
        
    elif func_type=='exp':
        out = np.exp(x)-1
        out[x <= 0]=0
    return(out)

##########################################################################
##########################################################################
#### helper functions for running the model (8/2/2019 - AL)

### assume even counts of all trial types - 8/2/2019 - AL
def return_cond_list(num_trials_total,num_trial_types,random_on=True):
    
    num_trials_condition = int(np.round(num_trials_total/num_trial_types))
    num_trial_total = num_trials_condition*num_trial_types

    for cond in np.arange(num_trial_types):

        if cond==0:
            condition_list = cond*np.ones(num_trials_condition)
        else:
            condition_list = np.append(condition_list,cond*np.ones(num_trials_condition))

    ### randomize order
    if random_on==True:
        condition_list = np.random.permutation(condition_list)
    return(condition_list,num_trial_total)

### updated - 3/2/2020 
### updated - 3/9/2020 - AL
def define_network_inputs(network_parm,input_dict):
    #### this is the most basic - 4 inputs - A,X, and C/C*
    if network_parm['model_type']=='rec':
        
        
        #### Define the inputs at each time point (7/30/2019 - AL)        
        num_tp = network_parm['num_stim']*network_parm['num_tp_stim']
        num_trial_types = 2*network_parm['num_stim']
        input_matrix = np.zeros((num_trial_types,network_parm['input_size'],num_tp))
        trial_types = {}
        
        ############
        trial_index = 0
        for stim0 in input_dict['stim0']:

            ##################################### 
            stim = 0
            current_tp_start = stim*network_parm['num_tp_stim']
            current_tp_end = current_tp_start + network_parm['num_tp_on']

            ### A or X
            input_matrix[trial_index:trial_index+len(input_dict['stim1']),:,current_tp_start:current_tp_end] = \
                                                                                input_dict['stim0'][stim0]

            for stim1 in input_dict['stim1']:

                #####################################                                                          
                stim = 1
                current_tp_start = stim*network_parm['num_tp_stim']
                current_tp_end = current_tp_start + network_parm['num_tp_on']
                ### C or C*
                input_matrix[trial_index,:,current_tp_start:current_tp_end] = input_dict['stim1'][stim1]
                
                trial_types[trial_index] = [stim0,stim1]

                trial_index += 1

    
    return(input_matrix,trial_types)


### updated 9/13/2019 - AL
def weight_array_generate(cell_type_counts,rand_std=.5,link_std=.5):
    cell_types_make = list(cell_type_counts.keys())
    ba = 0
    #print(cell_types_make)
    cell_type_list = []
    for cell_type in cell_types_make:
        
        #print(cell_type)

        count = cell_type_counts[cell_type]
        if count > 0:
            ba += 1
            for ci in np.arange(count):

                #a = 0 #np.random.uniform(.5,1) ### always have a > b (changed 8/29/2019)
                cell_select = cell_connect(cell_type,rand_std,link_std) ### shape should be options x input cells
                num_connect_types = cell_select.shape[0]

                cell_select_index = np.random.choice(np.arange(num_connect_types))

                if np.all([ci==0,ba==1]):
                    weight_array = np.expand_dims(cell_select[cell_select_index],1)
                else:
                    weight_array = np.append(weight_array,np.expand_dims(cell_select[cell_select_index],1),1)
                    
                cell_type_list.append(cell_type)
                
                
    ### mix up the random / none cells (9/18/2019 - AL)
    ### there is nothing that says the random connections 
    ### at each time step need to happen between the same set of cells
    ### so we mix up nones and randoms between time steps 
    cell_type_list = np.array(cell_type_list)
    mix_index = np.any(np.array([cell_type_list=='none',cell_type_list=='rand_sel',\
                     cell_type_list=='random_uni',cell_type_list=='random_norm']),0)

    ### mix up cells
    weight_array[0:2,mix_index] = weight_array[0:2,mix_index][:,np.random.permutation(sum(mix_index))]
    weight_array[2:4,mix_index] = weight_array[2:4,mix_index][:,np.random.permutation(sum(mix_index))]
    
    return(weight_array)

#### manually generate weights between input and population (hidden layer)
#### to create singles/ doubles - 8/2/2019 - AL
#### edited 9/13/2019 - AL (to add random cells)
#### cell types
def cell_connect(cell_type,rand_std,link_std):

    cell_special_on = True
    if cell_special_on:
        mean = (0,0)
        cov_ss = rand_std
        cov = [[link_std,cov_ss],[cov_ss,link_std]]
        x = abs(np.random.multivariate_normal(mean, cov, 1))[0]
        a= x[0]
        am = x[1]
        b = 0
    else:
        #b = 0 #an - a
        try1 = abs(np.random.normal(0,rand_std,2))
        a = np.max(try1)
        am = a
        b = np.min(try1)

    
    if cell_type=='single_0':
        cell_select = np.array([[a,b,b,b],[b,a,b,b],[a,b,a,a],[b,a,a,a]])
    
    elif cell_type=='single_1':
        cell_select = np.array([[b,b,a,b],[b,b,b,a],[a,a,b,a],[a,a,a,b]])
        #cell_select = np.array([[b,b,a,b],[b,b,b,a]])
        
    elif cell_type=='switch':
        cell_select = np.array([[a,b,b,am],[b,a,am,b]])
    elif cell_type=='stable':
        cell_select = np.array([[a,b,am,b],[b,a,b,am]])
    elif cell_type=='none':
        cell_select = np.array([[a,a,am,am],[b,b,b,b],[a,a,b,b],[b,b,am,am]])
        
    #### add in random options (9/13/2019 - AL)
    elif cell_type=='rand_sel':
        cell_select = np.array([[a,b,b,b],[b,a,b,b],[a,b,a,a],[b,a,a,a],\
                               [b,b,a,b],[b,b,b,a],[a,a,b,a],[a,a,a,b],\
                               [a,b,b,a],[b,a,a,b],[a,b,a,b],[b,a,b,a]])
    elif cell_type=='random_uni':
        cell_select = np.array([np.random.uniform(0,1,4)])
    elif cell_type=='random_norm':
        cell_select = np.array([abs(np.random.normal(0,rand_std,4))])
        
    return(cell_select)

#### first put in A/X selectivity based on W matrix - 3/18/2020 
#### updated 3/19/2020 - AL
###############################################
#### set up A/X selectivity ###################
#### IH weight matrix - 3/10/2020 - AL

#### focus on keeping total number of C/C* selectivity neurons CONSTANT! (note made - 3/16/2020 - Al)

### set up A/X selectivity - 
### num_trial_types,input_feature,num_tp = input_matrix.shape

def associate_IH(AX_sen,network_parm):
    
    IH = np.zeros((network_parm['input_size'],network_parm['N']))

    ### find neurons that are selective or A or X: 
    A_sel_index = AX_sen[0,:]>AX_sen[1,:] ### A sel
    X_sel_index = AX_sen[0,:]<AX_sen[1,:] ### X sel

    #print('number of A sel neurons: '+str(sum(A_sel_index)))
    #print('number of X sel neurons: '+str(sum(X_sel_index)))
    #print('total A/X sel: '+str(sum(np.any([A_sel_index,X_sel_index],0))))
    
    IH[0:2,:] = AX_sen
    ########################################################
    ### then add in C/C* selectivity based on link percent 
    #### updated - 3/18/2020 - AL

    #### test out different link percents - check C/C* sel
    lp = network_parm['link_percents']# = 0.95
    #print('current link percent: '+str(lp))

    link_A = int(np.round(lp*sum(A_sel_index)))
    link_X = int(np.round(lp*sum(X_sel_index)))

    #print('link with A: '+str(link_A))
    #print('link with X: '+str(link_X))

    #print('link with A/X: '+str(link_A+link_X))

    other_sel = network_parm['CCp_sel'] - (link_A+link_X) #CCp_sel_check

    #print('left over C/C* sel: '+str(other_sel))

    ######################################################
    ### set up AC link: 
    unlink_A = sum(A_sel_index) - link_A
    a_link = abs(np.random.normal(0,network_parm['link_std'],link_A))
    a_unlink = np.zeros(unlink_A)
    from_C_AC = np.append(a_link,a_unlink) ## these are the connections from C to neurons that prefer A (to create AC)
    IH[2,A_sel_index] = np.random.permutation(from_C_AC)

    ### set up XC* 8link
    unlink_X = sum(X_sel_index) - link_X
    x_link = abs(np.random.normal(0,network_parm['link_std'],link_X))
    x_unlink = np.zeros(unlink_X)
    from_Cp_XCp = np.append(x_link,x_unlink) ## these are the connections from C* to neurons that prefer X (to create XC*)
    IH[3,X_sel_index] = np.random.permutation(from_Cp_XCp)

    #############################################################

    ### find neurons that prefer neither A or X 
    ### put in C/Cp selectivity 

    ### 
    CCP = abs(np.random.normal(0,network_parm['link_std'],other_sel))
    to_CCp = np.append(CCP.reshape(-1,1),np.zeros(other_sel).reshape(-1,1),1)
    for index in np.arange(other_sel):
        to_CCp[index,:] = np.random.permutation(to_CCp[index,:])


    non_sel_index = return_true_index(AX_sen[0,:]==AX_sen[1,:])
    non_sel_index = np.array(np.random.permutation(non_sel_index))

    IH[2:4,non_sel_index[np.arange(other_sel)]] = to_CCp.T

    #### checking selectivity - 
    check_CCp_sel = sum(np.any([IH[2,:]>IH[3,:],IH[2,:]<IH[3,:]],0))
    #print('check CCp sel: '+str(check_CCp_sel))
    check_AX_sel =sum(np.any([IH[0,:]>IH[1,:],IH[0,:]<IH[1,:]],0))
    #print('check AX sel: '+str(check_AX_sel))
    
    return(IH)

##################################################################################
### Two layer model 
#### 7/30/2019 - AL
### updated 3/2/2020 - AL

class TL(nn.Module):
    def __init__(self, network_parm,IH,W):
        super(TL, self).__init__()

        if network_parm['model_type']=='rec':
            
            ### define connections from sensory to hidden layere
            self.IH = IH #np.random.normal()

            ### define dynamics over time
            self.W = W #np.random.normal()

        ### initialize the bias on RNN layer
        self.bias = np.zeros(network_parm['N'])
        
    def forward(self,Vt,network_parm,current_input):
    
        ### noise
        self.noise_hidden = np.random.normal(0,network_parm['noise_level'],network_parm['N'])    
        
        ### all inputs 
        u = np.dot(self.W,Vt) + self.noise_hidden + self.bias + np.dot(current_input,self.IH)
        
        dV = (1/network_parm['tau'])*(transfer_func(u,func_type=network_parm['transfer_func'])-Vt)

        Vt_out = Vt + dV
        return(Vt_out)

### this function randomly chooses test trials and returns both the index of test and train trials
### input - cond trials should be the index of the condition in question
### 7/31/2019 - AL
def return_test_train_cond(cond_trials,test_size):
    num_cond_trials = len(cond_trials)
    choose_trials_bool = np.append(np.ones(int(test_size)),np.zeros(int(num_cond_trials-test_size)))
    choose_trials_bool = np.random.permutation(choose_trials_bool)

    test_trials = cond_trials[choose_trials_bool==1]
    train_trials = cond_trials[choose_trials_bool==0]
    return(test_trials,train_trials)

def down_sample(condition_list,train_trial_all):
    ### downsample train set
    train_cond_list = condition_list[train_trial_all]
    train_cond,train_cond_counts = np.unique(train_cond_list,return_counts=True)
    ### check counts: 
    if len(np.unique(train_cond_counts))>1:
        min_counts = np.mean(train_cond_counts)
        ### downsample
        #print('downsample training so you have equal number of trials per condition')
        for cond in train_cond:
            down_train_cond = random.sample(train_trial_all[train_cond_list==cond],min_counts)
            if cond==train_cond[0]:
                all_down_train_cond = down_train_cond
            else:
                all_down_train_cond = np.append(all_down_train_cond,down_train_cond)

        train_trial_all_use = all_down_train_cond
    else:
        train_trial_all_use = np.copy(train_trial_all)
        
    return(train_trial_all_use)

### determine split of train test (8/2/2019 - AL)
def split_train_test(condition_list,test_percent_y,cond_train_labels):
    num_trials = len(condition_list)
    num_cond = len(cond_train_labels)
    
    test_size = num_trials*test_percent_y
    test_cond_size = np.round(test_size/num_cond)
    test_size = num_cond*test_cond_size #num_trial_types*test_cond_size

    all_trials = np.arange(num_trials)

    for cond in cond_train_labels:
        cond_trials = all_trials[condition_list==cond]
        test_trials,train_trials = return_test_train_cond(cond_trials,test_size)

        if cond==cond_train_labels[0]:
            test_trial_all = test_trials
            train_trial_all = train_trials
        else:
            test_trial_all = np.append(test_trial_all,test_trials)
            train_trial_all = np.append(train_trial_all,train_trials)

    #if run==0:
    #    print('held out test set counts: '+str(np.unique(condition_list[test_trial_all],return_counts=True)))

    ### downsample train set
    train_trial_all_use = down_sample(condition_list,train_trial_all)

    #if run==0:
    #    print('train set counts: '+str(np.unique(condition_list[train_trial_all_use],return_counts=True)))
    return(test_trial_all,train_trial_all_use)


#### return a binary version of the condition list based on trials we want to group together
#### see cond_comp_dict - 8/5/2019 - AL
#### - updated 4/2/2020 - AL - jjust made paradigm_type = 'AC_seq'

def return_bin_condition_list(condition_list,current_clf_name,cond_comp_dict,paradigm_type='AC_seq'):
    condition_list_bin = np.zeros(condition_list.shape)

    for bin_name in np.arange(2):
        bin_conds = cond_comp_dict[paradigm_type][current_clf_name][bin_name]
        for bc in bin_conds:

            if bc==bin_conds[0]:
                bin_list = np.expand_dims(condition_list==bc,1)
            else:
                bin_list = np.append(bin_list,np.expand_dims(condition_list==bc,1),1)

        condition_list_bin[np.any(bin_list,1)] = bin_name
    return(condition_list_bin)

#### train classifier and get accuracy with AUC on test set - 8/2/2019 - AL
#### make sure that condition_list is now in binary format (all conditions either 0 or 1)
def clf_train_test(test_trial_all,train_trial_all_use,clf_class,\
                   opt_params,condition_list_bin,response_all,tp_win=1):
    
    num_trials,num_cells,num_tp = response_all.shape
    num_tp_clf = num_tp-tp_win+1
    ############################
    all_tp_clf = {}
    all_tp_auc = np.zeros(num_tp_clf)
    for tp_train in np.arange(num_tp_clf):

        y_train = condition_list_bin[train_trial_all_use]
        X_train = np.mean(response_all[train_trial_all_use,:,tp_train:tp_train+tp_win],2)

        y_test = condition_list_bin[test_trial_all]
        X_test = np.mean(response_all[test_trial_all,:,tp_train:tp_train+tp_win],2)

        clf_c = clf_class(**opt_params).fit(X_train, y_train)
        all_tp_clf[tp_train] = clf_c
        
        y_score = clf_c.decision_function(X_test)
        #print(y_score.shape)
        fpr, tpr, _ = roc_curve(y_test, y_score,pos_label=1)
        all_tp_auc[tp_train] = auc(fpr, tpr)

    return(all_tp_clf,all_tp_auc)


#### zscore reponses (looking for categories of cells) - 7/31/2019 - AL
### add condition list binary (9/16/2019 - AL)
### updated - 4/3/2020 - AL
def zscore_activity(response_all,condition_list,condition_list_bin,network_parm,num_shuffles = 1000):
    
    num_trials,num_cells,num_tp = response_all.shape

    #### downsample conditions (based on full trial types) if necessay (7/31/2019)
    use_trials = down_sample(condition_list,np.arange(num_trials))
    response_use = response_all[use_trials]

    ### use binary condition labels 
    cond_list_use = condition_list_bin[use_trials]

    ### zscored firing rate differences 
    num_tp_win = network_parm['num_tp']-network_parm['tp_win_zscore']+1
    zscore_diff = np.zeros((num_tp_win,num_cells))
    
    for tp in np.arange(num_tp_win):
        
        cond0 = np.nanmean(response_use[cond_list_use==0,:,tp:tp+network_parm['tp_win_zscore']],2)
        cond1 = np.nanmean(response_use[cond_list_use==1,:,tp:tp+network_parm['tp_win_zscore']],2)
        cell_diff = np.mean(cond0,0)-np.mean(cond1,0)

        cell_diff_shuffle = np.zeros((num_shuffles,num_cells))
        
        #print('shuffle_test')
        for shuffle in np.arange(num_shuffles):
            shuffle_cond_list = np.random.permutation(np.copy(cond_list_use))
            cond0 = np.nanmean(response_use[shuffle_cond_list==0,:,tp:tp+network_parm['tp_win_zscore']],2)
            cond1 = np.nanmean(response_use[shuffle_cond_list==1,:,tp:tp+network_parm['tp_win_zscore']],2)
            cell_diff_shuffle[shuffle,:] = np.mean(cond0,0)-np.mean(cond1,0)

        #print(np.std(cell_diff_shuffle,0))
        std_hold = np.std(cell_diff_shuffle,0)
        mean_hold = np.mean(cell_diff_shuffle,0)
        
        if np.any(std_hold==0):
            zscore_diff[tp,std_hold==0] = (cell_diff[std_hold==0] - mean_hold[std_hold==0])
            zscore_diff[tp,std_hold!=0] = (cell_diff[std_hold!=0] - mean_hold[std_hold!=0])\
                                                                           /std_hold[std_hold!=0]
            print('zscore - std = zero')
            sys.stdout.flush()
        else:
            zscore_diff[tp,:] = (cell_diff - mean_hold)/std_hold
        
    return(zscore_diff)

    ##############################################################################
    
#### return the periods of zscore for the counts calculation 
### 4/3/2020 - Al
def return_zscore_blocks(zscore_diff,network_parm):
    stim_period_starts = np.arange(network_parm['num_stim'])*network_parm['num_tp_stim']
    stim_period_ends = stim_period_starts + network_parm['num_tp_on']

    zscore_diff_periods = np.zeros((network_parm['num_stim'],network_parm['N']))
    for stim in np.arange(network_parm['num_stim']):
        zscore_diff_periods[stim,:] = np.mean(zscore_diff[stim_period_starts[stim]:\
                                                       stim_period_ends[stim]],0)
        
    return(zscore_diff_periods)


#### create cell counts table - 7/31/2019 - AL
def create_cell_counts_table(zscore_diff):
    
    z_threshold = -ndtri(.05/2)
    cell_type_counts = {}
    cell_type_counts['AA'] = sum(np.all([zscore_diff[0,:]>=z_threshold,zscore_diff[1,:]>=z_threshold],0))
    cell_type_counts['0A'] = sum(np.all([abs(zscore_diff[0,:])<=z_threshold,zscore_diff[1,:]>=z_threshold],0))
    cell_type_counts['XA'] = sum(np.all([zscore_diff[0,:]<=-z_threshold,zscore_diff[1,:]>=z_threshold],0))

    cell_type_counts['A0'] = sum(np.all([zscore_diff[0,:]>=z_threshold,abs(zscore_diff[1,:])<=z_threshold],0))
    cell_type_counts['00'] = sum(np.all([abs(zscore_diff[0,:])<=z_threshold,abs(zscore_diff[1,:])<=z_threshold],0))
    cell_type_counts['X0'] = sum(np.all([zscore_diff[0,:]<=-z_threshold,abs(zscore_diff[1,:])<=z_threshold],0))

    cell_type_counts['AX'] = sum(np.all([zscore_diff[0,:]>=z_threshold,zscore_diff[1,:]<=-z_threshold],0))
    cell_type_counts['0X'] = sum(np.all([abs(zscore_diff[0,:])<=z_threshold,zscore_diff[1,:]<=-z_threshold],0))
    cell_type_counts['XX'] = sum(np.all([zscore_diff[0,:]<=-z_threshold,zscore_diff[1,:]<=-z_threshold],0))

    counts = [cell_type_counts['AA'],cell_type_counts['0A'],cell_type_counts['XA'],cell_type_counts['A0'],\
    cell_type_counts['00'],cell_type_counts['X0'],cell_type_counts['AX'],cell_type_counts['0X'],cell_type_counts['XX']]
    return(counts,cell_type_counts)


#### calculate angle
##### NOTE - SAMPLES X FEATURES (so it should be 2 x neurons) 8/29/2019 - AL
def cal_angle(data):
    cos_sim = cosine_similarity(data)
    angle = np.degrees(np.arccos(cos_sim[0][1]))
    return(angle,cos_sim[0][1])

### calculate the angle between axes/classifiers
### (8/30/2019 - AL)
### updated 4/6/2020 - AL
def cal_angle_axes(angle,all_clf,comp_compare,dict_axes,p_index,run):    
    for comp in comp_compare:

        clf_0 = all_clf[dict_axes[comp[0]]['name']][dict_axes[comp[0]]['tp']].coef_[0]
        clf_1 = all_clf[dict_axes[comp[1]]['name']][dict_axes[comp[1]]['tp']].coef_[0]

        data = np.append(clf_0.reshape(-1,1),clf_1.reshape(-1,1),1).T

        ### store the angle in the run comparison dictionary 
        angle['_'.join(comp)][p_index,run],cos_sim = cal_angle(data)
    return(angle)


#### To understand how interference and associations play together (8/29/2019)
#### calculate the cross auc (8/30/2019 - AL)
#### updated - 4/6/2020 - Al 

#### cross_auc[clf_train_name][test_name] --> update each time
#### clf_comp - clf/responses to compare
#### num_tp 
#### all_clf - current runs trained clfs
#### condition_list - conditions on this run
#### paradigm_type 
#### cond_comp_dict[paradigm_type][cond] - explains how for each paradigm type, trial types are broken down
#### test_trial_all - trial index to use
def cross_auc_clf(cross_auc,clf_comp,num_tp,all_clf,condition_list,\
                  cond_comp_dict,test_trial_all,p_index,run,network_parm,response_all):
    #### clf train
    for clf_type in clf_comp:

        #### clf train period
        for tp_train in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):

            clf_train_name = clf_type+str(tp_train)
            clf_c = all_clf[clf_type][tp_train]

            ##################################
            #### test response comparison type
            for current_cond_type in clf_comp:

                condition_list_bin = return_bin_condition_list(condition_list,current_cond_type,\
                                                           cond_comp_dict,paradigm_type='AC_seq')
                #### test response time period (tp)    
                for tp_test in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):

                    test_name = current_cond_type+str(tp_test)

                    y_test = condition_list_bin[test_trial_all]
                    X_test = np.mean(response_all[test_trial_all,:,tp_test:tp_test+network_parm['tp_win']],2)
                    y_score = clf_c.decision_function(X_test)
                    fpr, tpr, _ = roc_curve(y_test, y_score,pos_label=1)
                    cross_auc[clf_train_name][test_name][p_index,run] = auc(fpr, tpr)    
                    
    return(cross_auc)


#### POSTDICTION calculate how AX sen /AX mem responds during unexpected verses expected trials
#### (8/30/2019 - AL)
#### updated 4/6/2020 - AL
### TRIALS [0:AC,1:AC*,2:XC,3:XC*]

def postdiction_test(eu_auc,clf_comp,all_clf,exp_comp,exp_dict,condition_list,\
                     test_trial_all,response_all,tp_test,network_parm,cond_comp_dict,p_index,run):
    #tp_test = 1 ### only look at the test

    #### clf train
    for clf_type in clf_comp:

        #### clf train period
        for tp_train in np.arange(network_parm['num_tp']-network_parm['tp_win']+1):

            clf_train_name = clf_type+str(tp_train)
            clf_c = all_clf[clf_type][tp_train]

            ##################################
            #### test response comparison type
            for current_cond_type in clf_comp:

                for exp_comp in ['exp','unexp']:

                    test_name = current_cond_type+str(tp_test)+exp_comp

                    condition_list_bin = return_bin_condition_list(condition_list,current_cond_type,\
                                                           cond_comp_dict,paradigm_type='AC_seq')

                    ### only use conditions that are expected or unexpected 
                    current_cond_use = np.append(exp_dict[exp_comp][current_cond_type][0],\
                                                 exp_dict[exp_comp][current_cond_type][1])
                    for cc in current_cond_use:
                        tf_c = np.array(condition_list==cc).reshape(-1,1)
                        if cc == current_cond_use[0]:
                            tf = tf_c
                        else:
                            tf = np.append(tf,tf_c,1)
                    use_trials = np.any(tf,1)

                    #### trials from test_trial_all to use 
                    sub_test_trial_all = test_trial_all[use_trials[test_trial_all]]

                    y_test = condition_list_bin[sub_test_trial_all]
                    X_test = np.mean(response_all[sub_test_trial_all,:,tp_test:tp_test+network_parm['tp_win']],2)
                    y_score = clf_c.decision_function(X_test)
                    fpr, tpr, _ = roc_curve(y_test, y_score,pos_label=1)

                    eu_auc[clf_train_name][test_name][p_index,run] = auc(fpr, tpr)    
                    
    return(eu_auc)
