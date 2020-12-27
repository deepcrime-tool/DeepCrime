import os

import keras
import tensorflow as tf
from keras.datasets import mnist

import numpy as np
import csv
import h5py

from experimental_analysis.Model import MnistModel, MovieModel

output_dir = ''
model_dir = ''
redundancy_output_dir = ''
weak_ts_dir = ""
ts_size = 72601
model_num = 20

mutation_prefix_list = []

subject_name = 'movie_recomm'

model = MovieModel()


def analyse_redundancy():
    csv_file = redundancy_output_dir + subject_name + '_redundancy.csv'
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        for mutation_prefix_op1 in mutation_prefix_list:
            killing_info_op1 = np.load(output_dir + mutation_prefix_op1 + '_ki.npy')

            killing_probabilities_op1 = np.sum(killing_info_op1, axis=0) / len(killing_info_op1)
            num_zeros = len(np.where(killing_probabilities_op1 == 0)[0])
            for mutation_prefix_op2 in mutation_prefix_list:
                if mutation_prefix_op1 != mutation_prefix_op2:
                    print("Analysing " + str(mutation_prefix_op1) + " " + str(mutation_prefix_op2))
                    killing_info_op2 = np.load(output_dir + mutation_prefix_op2 + '_ki.npy')

                    killing_probabilities_op2 = np.sum(killing_info_op2, axis=0) / len(killing_info_op2)

                    comparison_array = np.less_equal(killing_probabilities_op1, killing_probabilities_op2)
                    np.save(redundancy_output_dir + mutation_prefix_op1 + '_' + mutation_prefix_op2 +'.npy', comparison_array)
                    num_true = round((len(np.where(comparison_array == True)[0]) - num_zeros) / (len(comparison_array) - num_zeros), 2)
                    writer.writerow([str(mutation_prefix_op1), str(mutation_prefix_op2), str(num_true)])


def analyse_triviality(model_dir):
    print("Predicting for Original")

    original_info_file = output_dir + 'original_prediction_info' + '.npy'
    if not (os.path.exists(original_info_file)):
        original_prediction_info = get_prediction_array(subject_name + '_original', model_dir)
        np.save(original_info_file, original_prediction_info)
    else:
        original_prediction_info = np.load(original_info_file)

    print("Predicting for Mutation")

    for mutation_prefix in mutation_prefix_list:
        print(mutation_prefix)
        if not (os.path.exists(output_dir + mutation_prefix + '_ki.npy')):
            mutation_info_file = output_dir + mutation_prefix + '.npy'
            if not (os.path.exists(mutation_info_file)):
                mutation_prediction_info = get_prediction_array(mutation_prefix, model_dir)
                np.save(mutation_info_file, mutation_prediction_info)
            else:
                mutation_prediction_info  = np.load(mutation_info_file)
            killing_info = get_killing_info(original_prediction_info, mutation_prediction_info, mutation_prefix)
        else:
            killing_info = np.load(output_dir + mutation_prefix + '_ki.npy')

        killing_probabilities = np.sum(killing_info, axis=0) / len(killing_info)
        print(killing_probabilities)
        range_value = 0.05

        csv_array = []
        csv_array.append(mutation_prefix)
        csv_array.append(str(len(np.where(killing_probabilities == 0)[0])))

        expected_value = np.sum(killing_probabilities)
        triviality_score = expected_value / ts_size

        print("Expected Value:" + str(expected_value))
        print("Triviality Score:" + str(triviality_score))

        for r in np.arange(0.05, 1.05, range_value):
            csv_array.append(len(np.where(np.logical_and(killing_probabilities > r - range_value, killing_probabilities <= r))[0]))

        csv_array.append(triviality_score)

        csv_file = output_dir + subject_name + '_triviality.csv'

        with open(csv_file, 'a') as f1:
            writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
            writer.writerow(csv_array)


def get_killing_info(original_prediction_info, mutation_prediction_info, mutation_prefix):
    killing_info = []

    for i in range(0, len(original_prediction_info)):
        killing_array = np.empty(len(original_prediction_info[i]))
        for j in range(0, len(original_prediction_info[i])):
            print()
            if (original_prediction_info[i][j] == True) and (mutation_prediction_info[i][j] == False):
                killing_array[j] = 1
            else:
                killing_array[j] = 0
        print(killing_array[j])
        killing_info.append(killing_array)

        #unique_label_list, unique_counts = np.unique(killing_array, return_counts=True, axis=0)

    killing_info = np.asarray(killing_info)
    np.save(output_dir + mutation_prefix + '_ki.npy', killing_info)

    return killing_info


def get_prediction_array(name_prefix, model_dir):
    mutation_list = []

    files = get_list_of_files_by_name(name_prefix, model_dir)
    #x_test, y_test, y_test_cl = model.get_test_data()
    #assert (len(files) == 20)
    prediction_info = []

    for i in range(0, model_num):
        file = model_dir + name_prefix + "_" + str(i) + ".h5"
        prediction_info.append(model.get_prediction_info(file))

    return prediction_info


def get_list_of_files_by_name(name_prefix, dir):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dir):
        for file in f:
            if name_prefix in file:
                files.append(os.path.join(r, file))

    #files = files.sort()    return files


def generate_deepmetis_weak_ts():
    for mutation_prefix in mutation_prefix_list:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        killing_info = np.load(output_dir + mutation_prefix + '_ki.npy')
        killing_probabilities = np.sum(killing_info, axis=0) / len(killing_info)

        num_of_killing = len(np.where(killing_probabilities != 0)[0])
        print("Number of killing:" + str(num_of_killing))

        num_to_remove = round(num_of_killing / 2)
        print("Removing " + str(num_to_remove))
        init_num_to_remove = num_to_remove
        indexes_list = []
        for r in (1.00, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05):
            indexes = np.where(killing_probabilities == r)[0]
            num_of_killing_with_value = len(indexes)
            print("Number of killing with probability " + str(r) + ":" + str(num_of_killing_with_value))
            if (num_of_killing_with_value > num_to_remove):
                num_to_delete = num_to_remove
                print(indexes)
                indexes = indexes[0:num_to_delete]
            else:
                num_to_delete =  num_of_killing_with_value

            print("Deleting " + str(num_to_delete) + " items from allowed " + str(num_to_remove))
            num_to_remove = num_to_remove - num_to_delete
            for element in indexes:
                indexes_list.append(element)
            print(indexes)

        y_test = np.delete(y_test, indexes_list)
        x_test = np.delete(x_test, indexes_list, axis = 0)
        assert (len(indexes_list) == init_num_to_remove)
        assert (len(y_test) + init_num_to_remove == 10000)
        print("Length now:" + str(len(y_test)))
        hf2 = h5py.File(weak_ts_dir + mutation_prefix + '.h5', 'w')
        hf2.create_dataset('x_test', data=x_test)
        hf2.create_dataset('y_test', data=y_test)
        hf2.close()


if __name__ == "__main__":
    analyse_triviality(model_dir)
    #analyse_redundancy()
    #generate_deepmetis_weak_ts()
