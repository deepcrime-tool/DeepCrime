#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import csv
import numpy as np
import h5py
import statsmodels.api as sm
#from scipy.stats import wilcoxon
from keras import backend as K
from mnist_mutate import mutate_M2, mutate_M4, mutate_M5
from patsy import dmatrices
import pandas as pd 

def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    return (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / np.sqrt(((nx-1)*np.std(orig_accuracy_list, ddof=1) ** 2 + (ny-1)*np.std(accuracy_list, ddof=1) ** 2) / dof)

def get_dataset(i):
    dataset_file = "/home/ubuntu/crossval_set_" + str(i) + ".h5"
    hf = h5py.File(dataset_file, 'r')
    xn_train = np.asarray(hf.get('xn_train'))
    xn_test = np.asarray(hf.get('xn_test'))
    yn_train = np.asarray(hf.get('yn_train'))
    yn_test = np.asarray(hf.get('yn_test'))

    return xn_train, yn_train, xn_test, yn_test

def train_and_get_accuracies(param, mutation):
    accuracy_list = range(0, 25)
    index = 0
    csv_file = "mnist_binary_search_" + str(mutation) + ".csv"
    
    with open(csv_file, 'a') as f1:
        writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
        for i in range(0, 5):
            x_train, y_train, x_test, y_test = get_dataset(i)
            for j in range(0, 5):
               print("Training " + str(index) + ", for param " + str(param))

               if (mutation == '2'):
                   accuracy, loss = mutate_M2(0, param, x_train, y_train, x_test, y_test, i, j)
               elif (mutation == '4r'):
                   accuracy, loss = mutate_M4(0, param, x_train, y_train, x_test, y_test, i, j, 1)
               elif (mutation == '4p'): 
                   accuracy, loss = mutate_M4(0, param, x_train, y_train, x_test, y_test, i, j, 0)
               elif (mutation == '5'): 
                   accuracy, loss = mutate_M5(param, x_train, y_train, x_test, y_test, i, j)
                   
               writer.writerow([str(i), str(j), str(param), str(accuracy), str(loss)])
               print("Loss " + str(loss) + ", Accuracy " + str(accuracy))
               accuracy_list[index] = accuracy
               index += 1
               K.clear_session()
       
    accuracy_dict[param] = accuracy_list
    return accuracy_list
                                                            
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold = 0.05):
    #w, p_value = wilcoxon(orig_accuracy_list, accuracy_list)
    
    list_length = len(orig_accuracy_list)
    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list
    
    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data) 
    
    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value = float(pv)    
    
    effect_size = cohen_d(orig_accuracy_list, accuracy_list)
    print("effect size:" + str(effect_size))
    is_sts = ((p_value < threshold) and effect_size > 0.2)
    print("p_value:" + str(p_value) + ", is_sts:" + str(is_sts))
    return is_sts

def get_accuracies(param, mutation):
    if (param in accuracy_dict):
        return accuracy_dict[param]
    else:
        return train_and_get_accuracies(param, mutation)
    
def get_orig_accuracies_from_file():
    accuracy_list = range(0, 25)
    
    with open(orig_result_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = -1
        
        for row in csv_reader:
            if line_count == -1:
                line_count += 1

            accuracy_list[line_count] = float(row['Accuracy'])
            line_count += 1

    return accuracy_list

def get_accuracies_from_file(num):
    percentages = (5, 10, 25, 50, 75, 80, 85, 90, 95, 99, 100)
    accuracyDict = {}
    line_count = -1   
    perc_index = 0
    
    with open(result_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = -1
        accuracy_list = range(0, num)
        for row in csv_reader:
            if line_count == -1:
                line_count += 1
                
            index = line_count % num
            
            if (index == 0):
                accuracy_list = range(0, num)
                
            accuracy_list[index] = float(row['Accuracy'])
            
            if (index == num - 1):
                accuracyDict[percentages[perc_index]] = accuracy_list
                perc_index += 1
    
            line_count += 1
        
    return accuracyDict 

def search_for_perfect(lower_bound, upper_bound, lower_accuracy_list, upper_accuracy_list, mutation):
    middle_bound = (upper_bound + lower_bound) / 2
    print(str(lower_bound) + " " + str(middle_bound) + " " + str(upper_bound))
    middle_accuracy_list = get_accuracies(middle_bound, mutation)
    
    is_sts = is_diff_sts(orig_accuracy_list, middle_accuracy_list)
    if (is_sts):
        upper_bound = middle_bound
        upper_accuracy_list = middle_accuracy_list
    else:
        lower_bound = middle_bound
        lower_accuracy_list = middle_accuracy_list
        
    if (upper_bound - lower_bound <= level_of_precision):
        if (is_sts):
            print("middle_bound:" + str(middle_bound))
            perfect = middle_bound
        else:
            print("middle_bound:" + str(upper_bound))
            perfect = upper_bound
        return perfect
    else:
        print("Changing interval to: [" + str(lower_bound) + ", " + str(upper_bound) + "]")
        search_for_perfect(lower_bound, upper_bound, lower_accuracy_list, upper_accuracy_list, mutation)
    
    print("pumpurum:" + str(lower_bound) + " " + str(upper_bound))    

lower_bound = 0
upper_bound = 100
level_of_precision = 5
num = 25

#mutations = ('4r', '4p', 'm5')
mutations = ('5')
orig_result_file_name = "mnist_orig_25.csv"
for mutation in mutations:
    print("Mutation:" + str(mutation))
    
    result_files_dir = "/home/ubuntu/"
    result_file_name = "mnist_mutation" + str(mutation) + ".csv"
    dataset_dir = "/home/ubuntu/"
    
    result_file = result_files_dir + result_file_name
    orig_result_file = result_files_dir + orig_result_file_name 
    accuracy_dict = get_accuracies_from_file(num)
    
    orig_accuracy_list = get_orig_accuracies_from_file()
    lower_accuracy_list = orig_accuracy_list
    #lower_accuracy_list = get_accuracies(lower_bound)
    if (mutation == '5'):
        upper_accuracy_list = get_accuracies(99, mutation)
    else:
        upper_accuracy_list = get_accuracies(upper_bound, mutation)
    
    #lower_killed = False
    lower_killed = is_diff_sts(lower_accuracy_list, orig_accuracy_list)
    upper_killed = is_diff_sts(orig_accuracy_list, upper_accuracy_list)
    
    if (lower_killed):
        print("The mutation is already killed at the lowest value:" + str(lower_bound))
    elif ((not lower_killed) and (not upper_killed)):
        print("The mutation is not killed in this given range of parameters: [" + str(lower_bound) + ", " + str(upper_bound) + "]")
    elif ((not lower_killed) and (upper_killed)):
        perfect = search_for_perfect(lower_bound, upper_bound, lower_accuracy_list, upper_accuracy_list, mutation)  
        print("perfect is:" + str(perfect))          