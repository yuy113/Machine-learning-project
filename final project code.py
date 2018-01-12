#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:50:00 2017

@author: yubingyao
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    "sklearn",
    "scipy",
    "numpy",
    "matplotlib"
]

test_requirements = []

setup(
    name='semisup_learn',
    version='0.0.1',
    description="Semisupervised Learning Framework",
    url='https://github.com/tmadl/semisup-learn',
    packages=[
        'methods', 'frameworks'
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='semisup-learn',
)

#the above is  Python setupcode for semisup-learn library
#semisup-learn library for semi-supervised learning including
Contrastive Pessimistic likelihood estimation(CPLE) framework

'''

####################################################################################################
import numpy as np

from CPLELearning import CPLELearningModel

# number of data points


############################################################################################
##active learning python library-libact

##############################################################################################
#!/usr/bin/env python3
"""
libact library for pooled based active learning algorithms.
the code and installation instruction in python:
   https://github.com/ntucllab/libact 
The detailed documetation:    http://libact.readthedocs.io/en/latest/index.html
This script simulates real world use of active learning algorithms. Which in the
start, there are only a small fraction of samples are labeled. During active
learing process active learning algorithm (QueryStrategy) will choose a sample
from unlabeled samples to ask the oracle to give this sample a label (Labeler).
In this example, ther dataset are from the digits dataset from sklearn. User
would have to label each sample choosed by QueryStrategy by hand. Human would
label each selected sample through InteractiveLabeler. Then we will compare the
performance of using UncertaintySampling and RandomSampling under
LogisticRegression.
"""

import copy
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling

##########################################################################################
'''
generate simulation scenarios for classification with binary and multi-class
#generate multi-class data using make_gaussian_quantiles and make_classification
#generate binary label class using make_gaussian_quantiles and make_classification
#introduce degree of noise of class labels in make_classification by flip_y=0.01,0.1,0.3,0.5
'''
###########################################################################################

from sklearn.datasets import make_classification,make_gaussian_quantiles
from sklearn.preprocessing import StandardScaler

#set up random seed for further simulation
np.random.seed(124)
'''
#label data-X_label,y_label for supervised classification
#mixture label and unlabeled dataset for semi-supervised classification
#python scikit-learn-LabelPropagation and LabelSpreading
#CPLE SVM in semisup_learn library in python
#randomly generate the datasets 100 times, randomly split the dataset into train and test set
#python scikit-learn-LabelPropagation and LabelSpreading
#output: classification error rates for label_propogation using RBF kernel with default values of other parameters and 
# labelSpread using KNN kernal using default value of the parameter-alpha=0.5 respectively
'''
def semi_prop_spread(X_train,y_train,X_test,y_test):
    from sklearn.semi_supervised import LabelPropagation
    from sklearn.semi_supervised import label_propagation
    label_prop_model = LabelPropagation()
    label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
    label_prop_model.fit(X_train,y_train)
    label_spread.fit(X_train,y_train)
    label_prop_pred=label_prop_model.predict(X_test)
    label_spread_pred=label_spread.predict(X_test)
    label_spread_error=np.sum(np.abs(label_spread_pred-y_test))/float(len(y_test))
    label_prop_error=np.sum(np.abs(label_prop_pred-y_test))/float(len(y_test))
    return label_prop_error,label_spread_error
'''
Compare supervised learning-SVM classification, semi-supervised version of SVM-S3VM 
and Hinted SVM in active learning
'''






'''
Active learning for classification:
  three basic components before training the model
  model(supervised classification algorithms in Scikit-learn)
  query_strategy-method to sample the unlabeled cases
  label-judge the label class of unlabeled cases by domain expert... 
  train the model on train dataset 
  predict on the test dataset
'''

#transform the simulated data to the dataset format for active learning in libact library
#the simulated dataset with train and test set with both with known class labeled 
#train set with mixture of small portion of labeled cases and mostly unlabeled cases
    # choose a dataset with unbalanced class instances
'''
split_train_test,transform the simulated data to the dataset format for active learning
take input
1. data with the format first element-the feature matrix,
second element-the class labels 
2. test_size, the percentage of test set in whole input data
3. size_label, the number of labeled cases in mixture train set
4. output_type:"semi"-the dataformat for semi-supervised learning
"act"-the data format for active learning
output: 1. mixture of labeled and unlabeled data in train set
2. test set with both feature matrix and class labeled to calculate the scores for classification
3. the fully labeled traing set with all cases perfectly labeled
'''
# libact classes
from libact.labelers import IdealLabeler
#randomly split into train and test dataset, 
#transform the fully labeled trained data into mixture of unlabeled and labeled data
def split_train_test(data,test_size,size_label,output_type):
    X = StandardScaler().fit_transform(data[0])
    Y = data[1]
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=test_size)
    fully_labeled_trn_ds = Dataset(X_trn, list(Y_trn))
    arr_tr = np.arange(len(Y_trn))
    # arr_te = np.arange(len(y_test))
    #idx_kfold=[int(i*len(train_y)/K) for i in np.arange(K+1)]
    np.random.shuffle(arr_tr)
    Y_trn=np.array(Y_trn,dtype=int)
    # np.random.shuffle(arr_te)
    X_train_label,y_train_label=X_trn[arr_tr[:size_label]],Y_trn[arr_tr[:size_label]]
    #  X_test_label,y_test_label=X_test[arr_te[:size_label],:],y_test[arr_te[:size_label]]
    X_train_unlabel,y_train_unlabel=X_trn[arr_tr[size_label:]],Y_trn[arr_tr[size_label:]]
    #   X_test_unlabel,y_test_unlabel=X_test[arr_te[size_label:],:],y_test[arr_te[size_label:]]
    while len(np.unique(y_train_label)) < len(np.unique(Y)):
            np.random.shuffle(arr_tr)
            X_train_label,y_train_label=X_trn[arr_tr[:size_label]],Y_trn[arr_tr[:size_label]]
            X_train_unlabel,y_train_unlabel=X_trn[arr_tr[size_label:]],Y_trn[arr_tr[size_label:]]
    if output_type=="act":         
      y_train_unlabel=[None]*len(y_train_unlabel) 
      Y_trn=y_train_label.tolist()+y_train_unlabel
      trn_ds = Dataset(X_trn,Y_trn)
      tst_ds = Dataset(X_tst, Y_tst.tolist())
      return trn_ds, tst_ds, fully_labeled_trn_ds
    elif output_type=="semi": 
      y_train_unlabel[:]=-1 
  #  y_test_unlabel[:]=-1         
   # X_test,y_test= np.vstack((X_test_label,X_test_unlabel)), np.hstack((
    #    y_test_label, y_test_unlabel))
      X_trn,Y_trn= np.vstack((X_train_label,X_train_unlabel)), np.hstack((
       y_train_label, y_train_unlabel))
      return X_trn,Y_trn,X_tst,Y_tst

#select the three components for active learning before training the model
#classification algorithms in Scikit-learn 
#query_strategy
#label-ideal-label assuming the sampled unlabeled cases all get perfect guess of label class

#QueryStrategy used for binary class
#RandomSampling, HintSVM,Query by Committee,density_weighted_uncertainty_sampling
from libact.query_strategies import QueryByCommittee,DWUS
#check the level of the noise by flip_y 0.01,0.1,0.3,0.5 in make_classification
flip_y_level=[0.01,0.1,0.3,0.5]
data_c_1 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=2, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[0])

data_c_2 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=2, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[1])

data_c_3 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=2, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[2])

data_c_4 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=2, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[3])


def accuracy_score_semisup1(data,label_size,test_size):
    from sklearn.metrics import accuracy_score
    from CPLELearning import CPLELearningModel
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    X = StandardScaler().fit_transform(data[0])
    y = data[1]
    result_sm={'E1': [], 'E2': [], 'E3': []}                  
    for i in range(10):
        X_trn,Y_trn,X_tst,Y_tst=split_train_test_Xy(X,y,test_size,label_size,output_type='semi')
        kernel="rbf"
        #supervised data 
        Xsupervised_tr = X_trn[Y_trn!=-1, :]
        ysupervised_tr = Y_trn[Y_trn!=-1]
        #supervised SVM
        model = SVC(kernel=kernel, probability=True)
        model2=LogisticRegression(C=0.1)
        model2.fit(Xsupervised_tr, ysupervised_tr)
        result_sm['E1'].append(accuracy_score(Y_tst,model2.predict(X_tst) ))
        #CPLE(pessimistic) SVM:
        model =  CPLELearningModel(model, predict_from_probabilities=True,max_iter=15000)
        model.fit(X_trn, Y_trn.astype(int))              
        result_sm['E2'].append(accuracy_score(Y_tst.astype(int),model.predict(X_tst)))
        #CPLE(optimistic) SVM
        CPLELearningModel.pessimistic = False
        model = CPLELearningModel(SVC(kernel=kernel, probability=True), predict_from_probabilities=True,max_iter=15000)
        model.fit(X_trn, Y_trn.astype(int))              
        result_sm['E3'].append(accuracy_score(Y_tst.astype(int),model.predict(X_tst)))
        #labelpropogation and labelspread
        #score_prop,score_spread=semi_prop_spread(X_trn,Y_trn,X_tst,Y_tst)
        #accuracy score for labelpropogation
        #result_sm['E4'].append(score_prop)
        #accuracy score for labelSpread
        #result_sm['E5'].append(score_spread)
    #average the 10 accuracy scores for each machine learning algorithm
    E_out_1 = np.mean(result_sm['E1'])
    E_out_2 = np.mean(result_sm['E2'])
    E_out_3 = np.mean(result_sm['E3'])
  #  E_out_4 = np.mean(result_sm['E4'])
   # E_out_5 = np.mean(result_sm['E5'])
    #E_out_6 = np.mean(result['E6'], axis=0)    
    return [E_out_1,E_out_2,E_out_3]  


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    C_in, C_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = trn_ds.data[ask_id]
        lb = lbr.label(X)
        trn_ds.update(ask_id, lb)
        model.train(trn_ds)
        #model.predict(tst_ds[0,:]), criterion='hamming', criterion='hamming'
        C_in = np.append(C_in, model.score(trn_ds))
        C_out = np.append(C_out, model.score(tst_ds))

    return C_in, C_out



def result_two_class_act(data,test_size,label_size,quota):
    result_tc = {'E1': [], 'E2': [], 'E3': [], 'E4': []}
    for i in range(10):  # repeat experiment
        trn_ds, tst_ds, fully_labeled_trn_ds=split_train_test(data,0.33,200,output_type='act')
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
       # trn_ds5 = copy.deepcopy(trn_ds)
        #trn_ds6 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model =  LogisticRegression(C=0.1)
        #model2=SVM(kernel='rbf')
        quota = 500 # number of samples to query
        #random sampling
        qs_rs=RandomSampling(trn_ds)

        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs_rs, quota)
        result_tc['E1'].append(E_out_1)

        #Density Weighted Uncertainty Sampling (DWUS) 
        qs_dwus=DWUS(trn_ds2)
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs_dwus, quota)
        result_tc['E2'].append(E_out_2)

        #Query by Committee sampling disagreement ='kl_divergence'
        qs_qbc_kl=QueryByCommittee(
                trn_ds3, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1)
                ],
              disagreement='kl_divergence'          
            )
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs_qbc_kl, quota)
        result_tc['E3'].append(E_out_3)

        #Query by Committee sampling disagreement ='vote'
        qs_qbc_vote=QueryByCommittee(
                trn_ds4, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1)
                ],
              disagreement='vote'          
            )
        _, E_out_4 = run(trn_ds4, tst_ds, lbr, model, qs_qbc_vote, quota)
    E_out_1 = np.mean(result_tc['E1'], axis=0)
    E_out_2 = np.mean(result_tc['E2'], axis=0)
    E_out_3 = np.mean(result_tc['E3'], axis=0)
    E_out_4 = np.mean(result_tc['E4'], axis=0)
  #  E_out_5 = np.mean(result_tc['E5'], axis=0)
   # E_out_6 = np.mean(result['E6'], axis=0)

    print("Random", E_out_1[::5].tolist())
    print("DensityWeightedUncertaintySampling ", E_out_2[::5].tolist())
    print("QueryByCommitteeByKLdivergence", E_out_3[::5].tolist())
    print("QueryByCommitteeByVote", E_out_4[::5].tolist())
    # print("VarianceReduction: ", E_out_5[::5].tolist())
    #print("BinaryMinimization: ", E_out_6[::5].tolist())
    query_num = np.arange(1, quota + 1)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    ax.plot(query_num, E_out_1, 'g', label='Random')
    ax.plot(query_num, E_out_2, 'k', label='DensityWeightedUncertaintySampling')
    ax.plot(query_num, E_out_3, 'r', label='QueryByCommitteeByKLdivergence')
    ax.plot(query_num, E_out_4, 'b', label='QueryByCommitteeByVote')
    #ax.plot(query_num, E_out_5, 'c', label='VarianceReduction')
    # ax.plot(query_num, E_out_6, 'm', label='BinaryMinimization')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy Score')
    plt.title('Experiment Result (Classification Accuracy Score)')
    plt.show()



#compare the four different active learning algorithms for binary class labels
#at different noise level of y
result_two_class_act(data_c_1,0.33,50,100)
result_two_class_act(data_c_2,0.33,50,100)
result_two_class_act(data_c_3,0.33,50,100)
result_two_class_act(data_c_4,0.33,50,100)



#compare semi-supervised SVM with supervised SVM
score_2_c_1=accuracy_score_semisup1(data_c_1,50,0.33)
score_2_c_2=accuracy_score_semisup1(data_c_2,50,0.33)
score_2_c_3=accuracy_score_semisup1(data_c_3,50,0.33)
score_2_c_4=accuracy_score_semisup1(data_c_4,50,0.33)

score_2_c_11=accuracy_score_semisup1(data_c_1,150,0.33)
score_2_c_21=accuracy_score_semisup1(data_c_2,150,0.33)
score_2_c_31=accuracy_score_semisup1(data_c_3,150,0.33)
score_2_c_41=accuracy_score_semisup1(data_c_4,150,0.33)
#line plot of two semi-supervised SVMs and supervised SVM with
#percentage of flipping class labels-0.01,0.1,0.3,0.5
perc_flip=[0.01,0.1,0.3,0.5]
values =np.vstack((score_2_c_1,score_2_c_2,score_2_c_3,score_2_c_4)).T
inds   =np.arange(len(perc_flip))
labels =["Supervised SVM","Pessimistic CPLE SVM","Optimistic CPLE SVM"]



#Plot a line graph
plt.figure(3, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.xticks(inds, perc_flip,rotation=45)
plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the second series in blue with square marker
plt.plot(inds,values[2,:],'dg--', linewidth=3) #Plot the third series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on

plt.ylabel("Classification Accuracy Score") #Y-axis label
plt.xlabel("Percentage of flipping well separated two-class labels") #X-axis label
plt.title("Accuracy Score vs Noisy levels of class labels \n 50 labeled instances") #Plot title
plt.xlim(0,3) #set x axis range
plt.ylim((np.min(values)-0.02),(np.max(values)+0.02)) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/testaccuracyscore_noisy50n_line_plot.pdf")
#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.close()


values2 =np.vstack((score_2_c_11,score_2_c_21,score_2_c_31,score_2_c_41)).T
inds   =np.arange(len(perc_flip))
labels =["Supervised SVM","Pessimistic CPLE SVM","Optimistic CPLE SVM"]


#Plot a line graph
plt.figure(3, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.xticks(inds, perc_flip,rotation=45)
plt.plot(inds,values2[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values2[1,:],'sb-', linewidth=3) #Plot the second series in blue with square marker
plt.plot(inds,values2[2,:],'dg--', linewidth=3) #Plot the third series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on

plt.ylabel("Classification Accuracy Score") #Y-axis label
plt.xlabel("Percentage of flipping well separated two-class labels") #X-axis label
plt.title("Accuracy Score vs Noisy levels of class labels\n 150 labeled instances") #Plot title
plt.xlim(0,3) #set x axis range
plt.ylim((np.min(values2)-0.02),(np.max(values2)+0.02)) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/testaccuracyscore_noisy150n_line_plot.pdf")
#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.close()







#QueryStrategy used for multi-class labels
#random sampling,Uncertainty Sampling,
def result_multi_class_act(data,test_size,label_size,quota):
    result_mc = {'E1': [], 'E2': [], 'E3': [], 'E4': [], 'E5': []}
    for i in range(10):  # repeat experiment
        trn_ds, tst_ds, fully_labeled_trn_ds=split_train_test(data,test_size,label_size,output_type='act')
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
        trn_ds5 = copy.deepcopy(trn_ds)
        #trn_ds6 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model =  LogisticRegression(C=0.1)
        
       # quota = 100  # number of samples to query
        #random sampling
        qs_rs=RandomSampling(trn_ds)

        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs_rs, quota)
        result_mc['E1'].append(E_out_1)

        #Expected Error Reduction(EER) active learning algorithm
        qs_eer = EER(trn_ds2, model=LogisticRegression(C=0.1))
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs_eer, quota)
        result_mc['E2'].append(E_out_2)

        #Uncertainty Sampling with least confidence active learning algorithm
        qs_us_lc=UncertaintySampling(
                trn_ds3, # Dataset object
                model=LogisticRegression(C=0.1),
                method='lc'
            )
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs_us_lc, quota)
        result_mc['E3'].append(E_out_3)

        #Uncertainty Sampling with smallest margin active learning algorithm
        #smallest margin (sm), it queries the instance whose posterior
        #probability gap between the most and the second probable labels is minimal;
        qs_us_sm=UncertaintySampling(
                trn_ds4, # Dataset object
                model=LogisticRegression(C=0.1),
                method= 'sm'
            )
        _, E_out_4 = run(trn_ds4, tst_ds, lbr, model, qs_us_sm, quota)
        result_mc['E4'].append(E_out_4)

        #Uncertainty Sampling with smallest entropy active learning algorithm
        #entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        #to be passed in as model parameter;
        qs_us_ent=UncertaintySampling(
                trn_ds5, # Dataset object
                model=LogisticRegression(C=0.1),
                method= 'entropy'
            )
        _, E_out_5 = run(trn_ds5, tst_ds, lbr, model, qs_us_ent, quota)
        result_mc['E5'].append(E_out_5)

    E_out_1 = np.mean(result_mc['E1'], axis=0)
    E_out_2 = np.mean(result_mc['E2'], axis=0)
    E_out_3 = np.mean(result_mc['E3'], axis=0)
    E_out_4 = np.mean(result_mc['E4'], axis=0)
    E_out_5 = np.mean(result_mc['E5'], axis=0)
   # E_out_6 = np.mean(result['E6'], axis=0)

    print("Random", E_out_1[::5].tolist())
    print("ExpectedErrorReduction(EER)", E_out_2[::5].tolist())
    print("UncertaintySamplingWithLeastConfidence: ", E_out_3[::5].tolist())
    print("UncertaintySamplingWithLeastConfidence: ", E_out_4[::5].tolist())
    print("UncertaintySamplingWithEntropy: ", E_out_5[::5].tolist())
#    print("BinaryMinimization: ", E_out_6[::5].tolist())
    query_num = np.arange(1, quota + 1)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    ax.plot(query_num, E_out_1, 'g', label='Random')
    ax.plot(query_num, E_out_2, 'k', label='EER')
    ax.plot(query_num, E_out_3, 'r', label='UncertaintySampling_lc')
    ax.plot(query_num, E_out_4, 'b', label='UncertaintySampling_sm')
    ax.plot(query_num, E_out_5, 'c', label='UncertaintySampling_entropy')
  #  ax.plot(query_num, E_out_6, 'm', label='BinaryMinimization')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy Score')
    plt.title('Experiment Result (Accuracy Score)')
    plt.show()

'''
Take the input of features data and class label data-X,y
randomly split into train and test with percentage of test set-test_size
transform the fully labeled train set into mixture of labeled and unlabeled set 
with size of labeled observations-label_size
fit supervised data in train set with supervised SVM with default parameter setup
fit CPLE(pessimistic) SVM and CPLE(optimistic) SVM in semilearn library
 
output: the accuracy score of supervised SVM and four semi-supervised learning algorithms

'''
def accuracy_score_semisup(X,y,label_size,test_size):
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    result_sm={'E1': [], 'E2': [], 'E3': []}                  
    for i in range(10):
        X_trn,Y_trn,X_tst,Y_tst=split_train_test_Xy(X,y,test_size,label_size,output_type='semi')
        kernel="rbf"
        #supervised data 
        Xsupervised_tr = X_trn[Y_trn!=-1, :]
        ysupervised_tr = Y_trn[Y_trn!=-1]
        #supervised SVM
        model = SVC(kernel=kernel, probability=True)
        model.fit(Xsupervised_tr, ysupervised_tr)
        result_sm['E1'].append(accuracy_score(Y_tst,model.predict(X_tst)))
        #CPLE(pessimistic) SVM:
        model =  CPLELearningModel(SVC(kernel=kernel, probability=True), predict_from_probabilities=True,max_iter=15000)
        model.fit(X_trn, Y_trn)              
        result_sm['E2'].append(accuracy_score(Y_tst,model.predict(X_tst)))
        #CPLE(optimistic) SVM
        CPLELearningModel.pessimistic = False
        model = CPLELearningModel(SVC(kernel=kernel, probability=True), predict_from_probabilities=True,max_iter=15000)
        model.fit(X_trn, Y_trn)              
        result_sm['E3'].append(accuracy_score(Y_tst,model.predict(X_tst)))
        #labelpropogation and labelspread
        #score_prop,score_spread=semi_prop_spread(X_trn,Y_trn,X_tst,Y_tst)
        #accuracy score for labelpropogation
        #result_sm['E4'].append(score_prop)
        #accuracy score for labelSpread
        #result_sm['E5'].append(score_spread)
    #average the 10 accuracy scores for each machine learning algorithm
    E_out_1 = np.mean(result_sm['E1'])
    E_out_2 = np.mean(result_sm['E2'])
    E_out_3 = np.mean(result_sm['E3'])
  #  E_out_4 = np.mean(result_sm['E4'])
   # E_out_5 = np.mean(result_sm['E5'])
    #E_out_6 = np.mean(result['E6'], axis=0)    
    return [E_out_1,E_out_2,E_out_3]  

#QueryStrategy used for multi-label class
#RandomSampling,MultilabelWithAuxiliaryLearner, MMC, BinaryMinimization
from libact.query_strategies import RandomSampling,UncertaintySampling
from libact.query_strategies.multiclass import EER
from libact.models import LogisticRegression, SVM
#dataset for multi-class labeled datasets
#using make_gaussian_quantiles and make_classification with n_classes=4 from scikit-learn library
cov_level=[0.5,1.0,2.0]
dat_gau_1=make_gaussian_quantiles(mean=None, cov=cov_level[0], n_samples=500, n_features=20, n_classes=4)
dat_gau_2=make_gaussian_quantiles(mean=None, cov=cov_level[1], n_samples=500, n_features=20, n_classes=4)
dat_gau_3=make_gaussian_quantiles(mean=None, cov=cov_level[2], n_samples=500, n_features=20, n_classes=4)
#compare the four different active learning algorithms for multi-class labels
result_multi_class_act(dat_gau_1,0.33,50,100)
result_multi_class_act(dat_gau_2,0.33,50,100)
result_multi_class_act(dat_gau_3,0.33,50,100)
#compare semi-supervised SVM with supervised SVM
score_cov05=accuracy_score_semisup1(dat_gau_1,200,0.33)
score_cov1=accuracy_score_semisup1(dat_gau_2,50,0.33)
score_cov2=accuracy_score_semisup1(dat_gau_3,50,0.33)
#line plot of semisupervised and supervised SVM with covariance level 0.5,1,2



#check the level of the noise by flip_y 0.01,0.1,0.3,0.5 in make_classification
flip_y_level=[0.01,0.1,0.3,0.5]

data_c_m_1 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=4, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[0])

data_c_m_2 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=4, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[1])

data_c_m_3 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=4, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[2])

data_c_m_4 = make_classification(n_samples=500, n_features=20, 
                           n_informative=16, n_redundant=4, n_classes=4, 
                           n_clusters_per_class=5,
                           flip_y=flip_y_level[3])
#compare among different active learning algorithms
result_multi_class_act(data_c_m_1,0.33,50,100)
result_multi_class_act(data_c_m_2,0.33,50,100)
result_multi_class_act(data_c_m_3,0.33,50,100)
result_multi_class_act(data_c_m_4,0.33,50,100)
#compare semi-supervised SVM with supervised SVM
X_m_1=data_c_m_1[0]
y_m_1=data_c_m_1[1]
score_noise1=accuracy_score_semisup(X_m_1.astype(float),y_m_1.tolist(),150,0.33)
score_noise2=accuracy_score_semisup1(data_c_m_2,50,0.33)
score_noise3=accuracy_score_semisup1(data_c_m_3,50,0.33)
score_noise4=accuracy_score_semisup1(data_c_m_4,50,0.33)


import pandas as pd
import numpy as np
#read the first breast cancer dataset
path='/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/'
with open(path+'wdbc.txt') as dat_c1:
    table = pd.read_table(dat_c1, sep=',', header=None,   lineterminator='\n')
dat_c1.shape()

dat_c1.columns

type(dat_c1)

dat_c1=pd.read_table(path+'wdbc.txt', sep=',', header=None,   lineterminator='\n')
dat_c1[0]

y_c=[]
for i in dat_c1[1]:
    if (i=='M'):
        y_c.append(1)
    elif(i=='B'):
        y_c.append(0)
dat_c1['y']=y_c
#dat_c1_xy=dat_c1[2:][:]

dat_c1.columns=[str(x) for x in range(32)]

names_xy=[str(x) for x in range(32)]+['y']

dat_c1.columns=names_xy


dat_c1_xy=dat_c1[names_xy[2:]]
#class label for first breast cancer dataset
#0: benign; 1: maglignant
dat_c1_y=dat_c1['y']
dat_c1_x=dat_c1_xy[names_xy[2:32]]
#numpy array for the features in first breast cancer dataset
x_dat_c1=dat_c1_x.values
np.shape(x_dat_c1)

dat_c12=pd.read_table(path+'wpbc.txt', sep=',', header=None,   lineterminator='\n')

names_dat12=[str(x) for x in range(len(dat_c12.columns)-1)]+['y']
y_c1=[]
for i in dat_c12[1]:
    if (i=='R'):
        y_c1.append(1)
    elif(i=='N'):
        y_c1.append(0)
dat_c12['y']=y_c1
dat_c12.columns=names_dat12
dat_c12[:5]
dat_c12_xy=dat_c12[names_dat12[2:]]
dat_c12_xy[:5]
#class label for first breast cancer dataset
#0: benign; 1: maglignant
dat_c12_y=dat_c12['y']
len(names_dat12)
dat_c12_x=dat_c12_xy[names_dat12[2:35]]
#numpy array for the features in first breast cancer dataset
x_dat_c12=dat_c12_x.values

np.shape(x_dat_c12)

#breast-cancer-wisconsindata
dat_c1=pd.read_table(path+'breast-cancer-wisconsindata.txt', sep=',', header=None,   lineterminator='\n')
'''
input argument for the function:
1. filename: the text file name wiht binary class label
2. label_1:the name in text file corresponding to label class 1
3. label_1:the name in text file corresponding to label class 0
read the text file and transform into the format with the output
1. the feature matrix with the format of numpy array type
2. the labeled class for each subject
'''

def dat_read_binary(filename,ncol_label,label_1,label_0):
    import pandas as pd
    path='/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/'
    dat_c=pd.read_table(path+filename, sep=',', header=None,   lineterminator='\n')
    y_c=[]
    for i in dat_c[ncol_label]:
      if (i==label_1):
        y_c.append(1)
      elif(i==label_0):
        y_c.append(0)
    dat_c['y']=y_c
    #new dataset contains only the features and the transformed class label 
    dat_c=dat_c.ix[:,(dat_c.columns!=ncol_label)]
    #dat_c=dat_c.ix[:,dat_c.columns!=0]
    names_dat=[str(x) for x in range(len(dat_c.columns)-1)]+['y']
    dat_c.columns=names_dat
    dat_c_xy=dat_c[names_dat[1:]]
    dat_c_x=dat_c_xy.ix[:,:(len(dat_c_xy.columns)-1)]
    #numpy array for the features in first breast cancer dataset
    x_dat_c=dat_c_x.values
    #the class label
    dat_c_y=dat_c['y']
    return x_dat_c,dat_c_y

#X_c12,y_c12=dat_read_binary('wpbc.txt',ncol_label=1,label_1='R',label_0='N')

X_c11,y_c11=dat_read_binary('wdbc.txt',ncol_label=1,label_1='M',label_0='B')

X_c13,y_c13=dat_read_binary('breast-cancer-wisconsindata.txt',ncol_label=10,label_1=4,label_0=2)

#transform the missing data ? to NA in numpy and delete the subject with missing features
x_5=[]
for i in X_c13[:,5]:
      if i=='?':
          x_5.append(np.nan)
      else:
          x_5.append(float(i))           
X_c13_new=np.column_stack((np.delete(X_c13, [5], axis=1),x_5))
#remove the observation with missing feature values
X_c13_new=X_c13_new[~np.isnan(X_c13_new.astype(float)).any(axis=1)]
y_c13=y_c13[~np.isnan(x_5)] 
#randomly split into train and test dataset, 
#transform the fully labeled trained data into mixture of unlabeled and labeled data
def split_train_test_Xy(X,y,test_size,size_label,output_type):
   #  X=X_c11
   # y=y_c11
    X = StandardScaler().fit_transform(X)
    Y = y
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=test_size)
    fully_labeled_trn_ds = Dataset(X_trn, list(Y_trn))
    arr_tr = np.arange(len(Y_trn))
    # arr_te = np.arange(len(y_test))
    #idx_kfold=[int(i*len(train_y)/K) for i in np.arange(K+1)]
    np.random.shuffle(arr_tr)
    # np.random.shuffle(arr_te)
    Y_trn=np.array(Y_trn,dtype=int)
    X_train_label,y_train_label=X_trn[arr_tr[:size_label]],Y_trn[arr_tr[:size_label]]
    #  X_test_label,y_test_label=X_test[arr_te[:size_label],:],y_test[arr_te[:size_label]]
    X_train_unlabel,y_train_unlabel=X_trn[arr_tr[size_label:]],Y_trn[arr_tr[size_label:]]
    #   X_test_unlabel,y_test_unlabel=X_test[arr_te[size_label:],:],y_test[arr_te[size_label:]]
    while len(np.unique(y_train_label)) < len(np.unique(Y)):
            np.random.shuffle(arr_tr)
            X_train_label,y_train_label=X_trn[arr_tr[:size_label]],Y_trn[arr_tr[:size_label]]
            X_train_unlabel,y_train_unlabel=X_trn[arr_tr[size_label:]],Y_trn[arr_tr[size_label:]]
    if output_type=="act":         
      y_train_unlabel=[None]*len(y_train_unlabel) 
      Y_trn=y_train_label.tolist()+y_train_unlabel
      trn_ds = Dataset(X_trn,Y_trn)
      tst_ds = Dataset(X_tst, Y_tst.tolist())
      return trn_ds, tst_ds, fully_labeled_trn_ds
    elif output_type=="semi": 
      y_train_unlabel[:]=-1 
  #  y_test_unlabel[:]=-1         
   # X_test,y_test= np.vstack((X_test_label,X_test_unlabel)), np.hstack((
    #    y_test_label, y_test_unlabel))
      X_trn,Y_trn= np.vstack((X_train_label,X_train_unlabel)), np.hstack((
       y_train_label, y_train_unlabel))
      return X_trn,Y_trn,X_tst,Y_tst



       
score_labelsize100_dat13=accuracy_score_semisup(X_c13_new.astype(float),y_c13.tolist(),100,0.33)

score_labelsize50_dat13=accuracy_score_semisup(X_c13_new.astype(float),y_c13,50,0.33)

score_labelsize150_dat13=accuracy_score_semisup(X_c13_new.astype(float),y_c13,150,0.33)

       
score_labelsize100_dat11=accuracy_score_semisup(X_c11.astype(float),y_c11,100,0.33)

score_labelsize50_dat11=accuracy_score_semisup(X_c11.astype(float),y_c11,50,0.33)

score_labelsize150_dat11=accuracy_score_semisup(X_c11.astype(float),y_c11,150,0.33)

#line plot of two semi-supervised SVMs and supervised SVM with
#percentage of flipping class labels-0.01,0.1,0.3,0.5
label_size_seq=[50,100,150]
values_d13 =np.vstack((score_labelsize50_dat13,score_labelsize100_dat13,score_labelsize150_dat13)).T
inds   =np.arange(len(label_size_seq))
labels =["Supervised SVM","Pessimistic CPLE SVM","Optimistic CPLE SVM"]


np.shape(values_d13)
#Plot a line graph
plt.figure(3, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.xticks(inds, label_size_seq,rotation=45)
plt.plot(inds,values_d13[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values_d13[1,:],'sb-', linewidth=3) #Plot the second series in blue with square marker
plt.plot(inds,values_d13[2,:],'dg--', linewidth=3) #Plot the third series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on

plt.ylabel("Classification Accuracy Score") #Y-axis label
plt.xlabel("Size of labeled instances") #X-axis label
plt.title("Accuracy Score vs Size of labeled instances \n wdbc real dataset") #Plot title
plt.xlim(0,2) #set x axis range
plt.ylim((np.min(values_d13)-0.02),(np.max(values_d13)+0.02)) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/testaccuracyscore_realdata13_line_plot.pdf")
#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.close()


values_d11 =np.vstack((score_labelsize50_dat11,score_labelsize100_dat11,score_labelsize150_dat11)).T
inds   =np.arange(len(label_size_seq))
labels =["Supervised SVM","Pessimistic CPLE SVM","Optimistic CPLE SVM"]


#Plot a line graph
plt.figure(3, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.xticks(inds, label_size_seq,rotation=45)
plt.plot(inds,values_d13[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
plt.plot(inds,values_d13[1,:],'sb-', linewidth=3) #Plot the second series in blue with square marker
plt.plot(inds,values_d13[2,:],'dg--', linewidth=3) #Plot the third series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on

plt.ylabel("Classification Accuracy Score") #Y-axis label
plt.xlabel("Size of labeled instances") #X-axis label
plt.title("Accuracy Score vs Size of labeled instances \n breast-cancer-wisconsin real dataset") #Plot title
plt.xlim(0,2) #set x axis range
plt.ylim((np.min(values_d11)-0.02),(np.max(values_d11)+0.02)) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/testaccuracyscore_realdata11_line_plot.pdf")
#Displays the plots.
#You must close the plot window for the code following each show()
#to continue to run
plt.show()
plt.close()


  
#the mixed labeled data format for semisupervised learning
X_trn_c11,Y_trn_c11,X_tst_c11,Y_tst_c11=split_train_test_Xy(X_c11,y_c11,0.33,50,output_type='semi')         
#the mixed labeled data format for semisupervised learning
X_trn_c13,Y_trn_c13,X_tst_c13,Y_tst_c13=split_train_test_Xy(X_c13_new.astype(float),y_c13,0.33,50,output_type='semi')         


#HAR data from UCI machine learning archive
path2='/Users/yubingyao/Documents/Course materials/CS589 Machine Learning/Final project/UCI HAR Dataset/'
X_train_har=np.loadtxt(path2+'train/X_train.txt', dtype=float)
y_train_har=np.loadtxt(path2+'train/y_train.txt', dtype=float)

np.shape(y_train_har)
#mixed labeled data format for semi-supervised learning
X_trn_c2,Y_trn_c2,X_tst_c2,Y_tst_c2=split_train_test_Xy(X_train_har,y_train_har,0.33,750,output_type='semi')         

X_test_har=np.loadtxt(path2+'test/X_test.txt', dtype=float)
y_test_har=np.loadtxt(path2+'test/y_test.txt', dtype=float)

X_har,y_har=np.vstack((X_train_har,X_test_har)), np.hstack((y_train_har, y_test_har))

def run2(trn_ds, tst_ds, lbr, model, qs, quota,step_size):
    C_in, C_out = [], []

    for _ in range(0,(quota+step_size),step_size):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = trn_ds.data[ask_id]
        lb = lbr.label(X)
        trn_ds.update(ask_id, lb)
        model.train(trn_ds)
        #model.predict(tst_ds[0,:]), criterion='hamming', criterion='hamming'
        C_in = np.append(C_in, model.score(trn_ds))
        C_out = np.append(C_out, model.score(tst_ds))

    return C_in, C_out



#perform active learning algorithms for 3 real datasets 
#active learning algorithms on first two datasets with binary labels 
def result_two_class_act1(X,y,test_size,label_size,quota,step_size):
    result_tc = {'E1': [], 'E2': [], 'E3': [], 'E4': [], 'E5': []}
    for i in range(10):  # repeat experiment
        trn_ds, tst_ds, fully_labeled_trn_ds=split_train_test_Xy(X,y,test_size,label_size,output_type='act')
        #trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
        #trn_ds5 = copy.deepcopy(trn_ds)
        #trn_ds6 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model =  LogisticRegression(C=0.1)
        #model2=SVM(kernel='rbf')
        #quota = 500 # number of samples to query
        #random sampling
        qs_rs=RandomSampling(trn_ds)

        _, E_out_1 = run2(trn_ds, tst_ds, lbr, model, qs_rs, quota,step_size)
        result_tc['E1'].append(E_out_1)

        #Density Weighted Uncertainty Sampling (DWUS) 
        #qs_dwus=DWUS(trn_ds2)
        #_, E_out_2 = run2(trn_ds2, tst_ds, lbr, model, qs_dwus, quota,step_size)
        #result_tc['E2'].append(E_out_2)

        #Query by Committee sampling disagreement ='kl_divergence'
        qs_qbc_kl=QueryByCommittee(
                trn_ds3, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1)
                ],
              disagreement='kl_divergence'          
            )
        _, E_out_3 = run2(trn_ds3, tst_ds, lbr, model, qs_qbc_kl, quota,step_size)
        result_tc['E3'].append(E_out_3)

        #Query by Committee sampling disagreement ='vote'
        qs_qbc_vote=QueryByCommittee(
                trn_ds4, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1)
                ],
              disagreement='vote'          
            )
        _, E_out_4 = run2(trn_ds4, tst_ds, lbr, model, qs_qbc_vote, quota,step_size)
        result_tc['E4'].append(E_out_4)    

    E_out_1 = np.mean(result_tc['E1'], axis=0)
    #E_out_2 = np.mean(result_tc['E2'], axis=0)
    E_out_3 = np.mean(result_tc['E3'], axis=0)
    E_out_4 = np.mean(result_tc['E4'], axis=0)
  #  E_out_5 = np.mean(result_tc['E5'], axis=0)
   # E_out_6 = np.mean(result['E6'], axis=0)

    print("Random", E_out_1[::5].tolist())
   # print("DensityWeightedUncertaintySampling ", E_out_2[::5].tolist())
    print("QueryByCommitteeByKLdivergence", E_out_3[::5].tolist())
    print("QueryByCommitteeByVote", E_out_4[::5].tolist())
    # print("VarianceReduction: ", E_out_5[::5].tolist())
    #print("BinaryMinimization: ", E_out_6[::5].tolist())
    query_num = np.arange(1, quota + 1+step_size,step_size)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    ax.plot(query_num, E_out_1, 'g', label='Random')
    #ax.plot(query_num, E_out_2, 'k', label='DensityWeightedUncertaintySampling')
    ax.plot(query_num, E_out_3, 'r', label='QueryByCommitteeByKLdivergence')
    ax.plot(query_num, E_out_4, 'b', label='QueryByCommitteeByVote')
    #ax.plot(query_num, E_out_5, 'c', label='VarianceReduction')
    # ax.plot(query_num, E_out_6, 'm', label='BinaryMinimization')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy Score')
    plt.title('Experiment Result (Classification Accuracy Score)')
    plt.show()


result_two_class_act1(X_c11.astype(float),y_c11,0.33,100,200,5)

result_two_class_act1(X_c13_new.astype(float),y_c13,0.33,100,200,5)


#active learning algorithms on third dataset with multi-class labels 
def result_multi_class_act1(X,y,test_size,label_size,quota,step_size):
    result_mc = {'E1': [], 'E2': [], 'E3': [], 'E4': [], 'E5': []}
    for i in range(10):  # repeat experiment
        trn_ds, tst_ds, fully_labeled_trn_ds=split_train_test_Xy(X,y,test_size,label_size,output_type='act')
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
        trn_ds5 = copy.deepcopy(trn_ds)
        #trn_ds6 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model =  LogisticRegression(C=0.1)
        #quota = 100  # number of samples to query
        #random sampling
        qs_rs=RandomSampling(trn_ds)

        _, E_out_1 = run2(trn_ds, tst_ds, lbr, model, qs_rs, quota,step_size)
        result_mc['E1'].append(E_out_1)

        #Expected Error Reduction(EER) active learning algorithm
        qs_eer = EER(trn_ds2, model=LogisticRegression(C=0.1))
        _, E_out_2 = run2(trn_ds2, tst_ds, lbr, model, qs_eer, quota,step_size)
        result_mc['E2'].append(E_out_2)

        #Uncertainty Sampling with least confidence active learning algorithm
        qs_us_lc=UncertaintySampling(
                trn_ds3, # Dataset object
                model=LogisticRegression(C=0.1),
                method='lc'
            )
        _, E_out_3 = run2(trn_ds3, tst_ds, lbr, model, qs_us_lc, quota,step_size)
        result_mc['E3'].append(E_out_3)

        #Uncertainty Sampling with smallest margin active learning algorithm
        #smallest margin (sm), it queries the instance whose posterior
        #probability gap between the most and the second probable labels is minimal;
        qs_us_sm=UncertaintySampling(
                trn_ds4, # Dataset object
                model=LogisticRegression(C=0.1),
                method= 'sm'
            )
        _, E_out_4 = run2(trn_ds4, tst_ds, lbr, model, qs_us_sm, quota,step_size)
        result_mc['E4'].append(E_out_4)

        #Uncertainty Sampling with smallest entropy active learning algorithm
        #entropy, requires :py:class:`libact.base.interfaces.ProbabilisticModel`
        #to be passed in as model parameter;
        qs_us_ent=UncertaintySampling(
                trn_ds5, # Dataset object
                model=LogisticRegression(C=0.1),
                method= 'entropy'
            )
        _, E_out_5 = run2(trn_ds5, tst_ds, lbr, model, qs_us_ent, quota,step_size)
        result_mc['E5'].append(E_out_5)

    E_out_1 = np.mean(result_mc['E1'], axis=0)
    E_out_2 = np.mean(result_mc['E2'], axis=0)
    E_out_3 = np.mean(result_mc['E3'], axis=0)
    E_out_4 = np.mean(result_mc['E4'], axis=0)
    E_out_5 = np.mean(result_mc['E5'], axis=0)
   # E_out_6 = np.mean(result['E6'], axis=0)

    print("Random", E_out_1[::5].tolist())
    print("ExpectedErrorReduction(EER)", E_out_2[::5].tolist())
    print("UncertaintySamplingWithLeastConfidence: ", E_out_3[::5].tolist())
    print("UncertaintySamplingWithLeastConfidence: ", E_out_4[::5].tolist())
    print("UncertaintySamplingWithEntropy: ", E_out_5[::5].tolist())
    #print("BinaryMinimization: ", E_out_6[::5].tolist())
    query_num = np.arange(1, quota + 1+step_size,step_size)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    ax.plot(query_num, E_out_1, 'g', label='Random')
    ax.plot(query_num, E_out_2, 'k', label='EER')
    ax.plot(query_num, E_out_3, 'r', label='UncertaintySampling_lc')
    ax.plot(query_num, E_out_4, 'b', label='UncertaintySampling_sm')
    ax.plot(query_num, E_out_5, 'c', label='UncertaintySampling_entropy')
    #ax.plot(query_num, E_out_6, 'm', label='BinaryMinimization')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('Number of Queries')
    plt.ylabel('Accuracy Score')
    plt.title('Experiment Result (Accuracy Score)')
    plt.show()
#randomly sample 1000 pairs of features and class labels each time
arr_har = np.arange(len(y_har))
np.random.shuffle(arr_har) 
X_har_sub,y_har_sub=X_har[arr_har[0:1000]],y_har[arr_har[0:1000]]
#comparisons among 5 active learning algorithms in randomly sampled subset with 500 observations   
result_multi_class_act1(X_har_sub,y_har_sub,0.33,100,100,5)

score_labelsize200_dat2=accuracy_score_semisup(X_har_sub.astype(float),y_har_sub.tolist(),200,0.33)

score_labelsize100_dat2=accuracy_score_semisup(X_har_sub,y_har_sub,100,0.33)

score_labelsize300_dat2=accuracy_score_semisup(X_har_sub.astype(float),y_har_sub,300,0.33)


