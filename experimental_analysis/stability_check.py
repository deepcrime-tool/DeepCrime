import glob
import csv
import os
import numpy as np


def get_accuracy_array_from_file(filename, row_index):
    accuracy = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if any(x.strip() for x in row):
                accuracy.append(row[row_index])

    return np.asarray(accuracy).astype(np.float32)

# error_mu > error_orig and rel_std > 0.05 and eff_size < 2 and rel_error_diff > 0.05


def check_stability(model_dir):
    original_file = root_dir + name_prefix + '.csv'
    original_accuracy = get_accuracy_array_from_file(original_file, 2)

    print(original_file)
    # print(original_accuracy)

    orig_std = np.std(original_accuracy) / np.mean(original_accuracy)
    orig_err = orig_std / np.sqrt(run_num)

    bad_cases = 0
    file_num = 0
    mut_std_list = []
    files = glob.glob(model_dir + "*", recursive=True)
    print(len(files))
    files = sorted(files)
    for filename in files:
        file_num = file_num + 1
        if not os.path.isdir(filename) and filename != original_file:
            # print(filename)
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            mut_std = np.std(mutation_accuracy) / np.mean(mutation_accuracy)
            mut_std_list.append(mut_std)
            mut_err = mut_std / np.sqrt(run_num)

            relative_difference = abs(mut_err - orig_err) / orig_err

            # if mut_std >= 0.05:
            if mut_err > orig_err and mut_std > 0.05 and relative_difference > 0.05:
                bad_cases = bad_cases + 1
                print(filename)
                # print(mutation_accuracy)
                print("orig_err:" + str(orig_err))
                print("mut_err:" + str(mut_err))
                print("orig_std:" + str(orig_std))
                print("mut_std:" + str(mut_std))
                print("relative difference:" + str(relative_difference))
                print('')
            else:
                print(filename)
                print("OKOKOKOKOK")

    print(str(bad_cases) + ' out of ' + str(file_num))
    print(np.mean(np.asarray(mut_std_list).astype(np.float32)))

if __name__ == "__main__":
    root_dir = '/DeepCrime/results/train/original/'
    model_dir = '/DeepCrime/results/train/accs/'
    name_prefix = 'lenet'

    run_num = 20
    print(np.sqrt(20))
    threshold = 0.05
    check_stability(model_dir)