import glob
import csv
import numpy as np
import collections
from stats import power, cohen_d


def get_overall_mutation_score(stats_dir, train_stats_dir):
    mutation_score = 0
    operator_num = 0
    excluded_num = 0
    for filename in glob.glob(stats_dir + "*"):
        print(filename)
        if '.csv' in filename:
            print('fff:' + filename)
            if is_binary_search_operator(filename):
                test_score, ins_score = get_binary_search_operator_mutation_score(filename)
            else:
                test_score, ins_score = get_exhaustive_operator_mutation_score(filename, train_stats_dir)

            # operator_num = operator_num + 1
            # mutation_score = mutation_score + test_score
            if ins_score == 0 and test_score != -1:
                operator_num = operator_num + 1
                mutation_score = mutation_score + test_score
            else:
                excluded_num = excluded_num +  1
            print("operator_num:" + str(operator_num))

    print("operator_num:" + str(operator_num))
    print("excluded_num:" + str(excluded_num))
    print("mutation_score:" + str(mutation_score))
    mutation_score = round(mutation_score / operator_num, 4)
    print('Overall Mutation Score:' + str(mutation_score*100))


def get_binary_search_operator_mutation_score(filename):
    file_short_name = get_file_short_name(filename)
    train_killed_conf = get_killed_conf(train_stats_dir + file_short_name)
    test_killed_conf = get_killed_conf(filename)


    if train_killed_conf == -1:
        print("not killed mutant detected")
        return 0, 10

    if test_killed_conf == -1:
        test_killed_conf = get_upper_bound(file_short_name)

    print("Train Killed Conf:" + str(train_killed_conf))
    print("Test Killed Conf:" + str(test_killed_conf))

    upper_bound = get_upper_bound(file_short_name)
    print("Upper Bound:" + str(upper_bound))
    if train_killed_conf == test_killed_conf:
        print('equal')
        mutation_score = 1
    elif upper_bound == train_killed_conf:
        mutation_score = -1
    else:
        mutation_score = round((upper_bound - test_killed_conf) / (upper_bound - train_killed_conf), 2)

    test_power_dict, ins_score_min, ins_score_max = get_power_dict_binary(accuracy_dir, filename, train_killed_conf, test_killed_conf, upper_bound)
    print('test_power_dict:' + str(test_power_dict))
    if ins_score_min > 0:
        print('ins_score_min:' + str(ins_score_min))
    if ins_score_max > 0:
        print('ins_score_max:' + str(ins_score_max))

    train_power_dict, ins_score_min_t, ins_score_max_t = get_power_dict_binary(train_accuracy_dir, filename, train_killed_conf, test_killed_conf, upper_bound)
    print('train_power_dict:' + str(train_power_dict))
    if mutation_score > 1:
        mutation_score = 1
    print('Mutation Score:' + str(mutation_score))
    print("Stability Score is:")
    print(abs(ins_score_max) + abs(ins_score_min))

    return mutation_score, abs(ins_score_max) + abs(ins_score_min)


def get_file_short_name(filename):
    return filename[filename.rindex("/") + 1:len(filename)]


def get_upper_bound(file_short_name):
    if 'delete_td' in file_short_name:
        return 99

    if 'change_learning_rate' in file_short_name:
        return lower_lr

    if 'change_epochs' in file_short_name or 'change_patience' in file_short_name:
        return 1

    return 100


def get_power_dict_binary(accuracy_dir, stats_file_name, train_killed_conf, test_killed_conf, upper_bound):
    original_file = accuracy_dir + prefix + '.csv'
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name)
    overall_num = 0
    unstable_num = 0
    dict_for_binary = {}
    for filename in glob.glob(accuracy_dir + "*"):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            #TODO: change digit replacement to regular expression
            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                    'False_', '').replace('_0', '').replace('_3', '').replace('_9', '').replace('_1', '')

            if (pow >= 0.8):
                dict_for_binary[float(mutation_parameter)] = 's'
            else:
                dict_for_binary[float(mutation_parameter)] = 'uns'

            dict_for_binary = collections.OrderedDict(sorted(dict_for_binary.items()))

    ins_score_min, ins_score_max = get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf, upper_bound)
    return dict_for_binary, ins_score_min, ins_score_max


def get_power_dict_exh(accuracy_dir, stats_file_name):
    original_file = accuracy_dir + prefix + '.csv'
    original_accuracy = get_accuracy_array_from_file(original_file, 2)
    name = get_replacement_name(stats_file_name)

    overall_num = 0
    unstable_num = 0
    dict_for_exh = {}
    for filename in glob.glob(accuracy_dir + "*"):
        if name in filename:
            mutation_accuracy = get_accuracy_array_from_file(filename, 2)
            pow = power(original_accuracy, mutation_accuracy)

            mutation_parameter = filename.replace(accuracy_dir, '').replace('.csv', '').replace(name + '_', '').replace(
                    'False_', '').replace('_0', '').replace('_3', '').replace('_9', '').replace('_1', '')
            #print('mutation_parameter:' + str(mutation_parameter))
            if (pow >= 0.8):
                dict_for_exh[mutation_parameter] = 's'
            else:
                dict_for_exh[mutation_parameter] = 'uns'

    return dict_for_exh


def get_ins_score(stats_file_name, dict_for_binary, train_killed_conf, test_killed_conf, upper_bound):
    found_first_stable = False
    unstable = 0
    stable = 200
    for key in dict_for_binary:
        if dict_for_binary[key] == 'uns':
            unstable = float(key)
        elif dict_for_binary[key] == 's' and not found_first_stable and float(key) >= test_killed_conf:
            stable = float(key)
            found_first_stable = True

    print('stable:' + str(stable))
    print('unstable:' + str(unstable))

    if stable == 200 or (unstable > stable and not('change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name )):
        return 1, 1

    if stable < unstable and train_killed_conf < unstable and ('change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if unstable < train_killed_conf and not('change_epochs' in stats_file_name or 'change_learning_rate' in stats_file_name or 'change_patience' in stats_file_name):
        return 0, 0

    if upper_bound - train_killed_conf == 0  or unstable == 0:
        return 0, 0

    if unstable == upper_bound:
        return 1, 1

    if 'change_epochs' in stats_file_name or 'change_patience' in stats_file_name:
        upper_bound = 1

    print(upper_bound)
    ins_score_min = round(abs(unstable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    ins_score_max = round(abs(stable - train_killed_conf) / abs(upper_bound - train_killed_conf), 2)
    return ins_score_min, ins_score_max


def get_accuracy_array_from_file(filename, row_index):
    accuracy = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if any(x.strip() for x in row):
                accuracy.append(row[row_index])

    return np.asarray(accuracy).astype(np.float32)


def get_killed_conf(filename):
    killed_conf = -1

    row_num = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print('row:' + str(row))
            # print(row[len(row) - 1])
            killed_conf = row[0]

            row_num = row_num + 1

    if row_num == 1:
        return -1

    if killed_conf != -1 and train_stats_dir in filename:
        file_short_name = filename[filename.rindex("/") + 1:len(filename)]
        killed_name_list.append(file_short_name + '_' + killed_conf)

    return float(killed_conf)


def get_exhaustive_operator_mutation_score(filename, train_stats_dir):
    print(filename)

    power_dict_exh_train = get_power_dict_exh(train_accuracy_dir, filename)
    print('power_dict_exh train:' + str(power_dict_exh_train))

    power_dict_exh_test = get_power_dict_exh(accuracy_dir, filename)
    print('power_dict_exh test:' + str(power_dict_exh_test))

    file_short_name = filename[filename.rindex("/") + 1:len(filename)]
    train_killed_conf = get_killed_from_csv(train_stats_dir + file_short_name)
    if len(train_killed_conf) == 0:
        return -1, -1

    test_killed_conf = get_killed_from_csv(filename)

    print('Train Killed Conf Length:' + str(len(train_killed_conf)))
    for killed_conf in train_killed_conf:

        if power_dict_exh_train.get(killed_conf) == 'uns':
            print('unstable kc:' + str(killed_conf))
            train_killed_conf.remove(killed_conf)

    print(train_killed_conf)
    print(test_killed_conf)

    killed_conf = np.intersect1d(train_killed_conf, test_killed_conf)
    print(killed_conf)

    if len(train_killed_conf) == 0:
        mutation_score = 0
    else:
        mutation_score = round(len(killed_conf) / len(train_killed_conf),  2)

    if not len(killed_conf) == 0:
        ins_score = get_ins_score_exh(killed_conf, power_dict_exh_test)
    else:
        ins_score = 0

    if ins_score > 0:
        print('ins_score:' + str(ins_score))
    else:
        print("It is ok")
    print('Mutation Score:' + str(mutation_score))
    return mutation_score, ins_score


def get_ins_score_exh(killed_conf, power_dict_exh_test):
    ins_num = 0
    for kc in killed_conf:
        if power_dict_exh_test.get(kc) == 'uns':
            ins_num = ins_num + 1

    return round(ins_num / len(killed_conf), 2)


def get_killed_from_csv(filename):
    index = get_outcome_row_index(filename)

    killed_conf = []
    killed_count = 0
    row_count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #print(row)
            #print(index)
            if row[index] == 'TRUE' or row[index] == 'True':
                if train_stats_dir in filename:
                    file_short_name = filename[filename.rindex("/") + 1:len(filename)]
                    killed_name_list.append(file_short_name + '_' + row[0])

                killed_count = killed_count + 1
                #if index == 3:
                killed_conf.append(row[0])
            row_count = row_count + 1

    ratio = round(killed_count / row_count, 2)
    print('For ' + filename + ' killed ' + str(killed_count) + ' out of ' + str(row_count))
    print('Ratio:' + str(ratio))

    return killed_conf


def get_outcome_row_index(filename):
    if ('disable_batching' in filename) or ('remove_validation_set' in filename):
        return 2
    else:
        return 3


def is_binary_search_operator(filename):
    operator_list = ['change_label', 'delete_td', 'unbalance_td', 'add_noise',
                     'output_classes_overlap', 'change_epochs', 'change_learning_rate', 'change_patience']
    for operator in operator_list:
        if operator in filename:
            return True

    return False


def get_replacement_name(stats_file_name):
    killed_mutation = stats_file_name.replace(stats_dir, prefix + '_')
    killed_mutation = killed_mutation.replace('_exssearch.csv', '_mutated0_MP')
    if 'change_epochs' in killed_mutation:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_False')
    else:
        killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP')

    killed_mutation = killed_mutation.replace('_nosearch.csv', '_mutated0_MP')
    killed_mutation = killed_mutation.replace('unbalance_td', 'unbalance_train_data')
    killed_mutation = killed_mutation.replace('delete_td', 'delete_training_data')
    killed_mutation = killed_mutation.replace('output_classes_overlap', 'make_output_classes_overlap')
    killed_mutation = killed_mutation.replace('change_patience', 'change_earlystopping_patience')
    return killed_mutation


def postprocess_killed_name_list():
    index = 0
    for killed_mutation in killed_name_list:
        killed_mutation = killed_mutation.replace('_exssearch.csv', '_mutated0_MP')
        if 'change_epochs' in killed_mutation:
            killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP_False')
        else:
            killed_mutation = killed_mutation.replace('_binarysearch.csv', '_mutated0_MP')
        killed_mutation = killed_mutation.replace('_nosearch.csv', '_mutated0_MP')
        killed_mutation = killed_mutation.replace('unbalance_td', 'unbalance_train_data')
        killed_mutation = killed_mutation.replace('delete_td', 'delete_training_data')
        killed_mutation = killed_mutation.replace('output_classes_overlap', 'make_output_classes_overlap')
        killed_mutation = prefix + '_' + killed_mutation
        print(killed_mutation)
        killed_name_list[index] = killed_mutation
        index = index  +  1

    return killed_name_list


if __name__ == "__main__":
    # epochs = 12
    # lower_lr = 0.001
    # upper_lr = 1
    # prefix = 'mnist'
    # accuracy_dir = '/mnist/results_easy/'
    # train_accuracy_dir = '/mnist/results_train/'

    # epochs = 5
    # lower_lr = 0.0001
    # upper_lr = 0.001
    # prefix = 'movie_recomm'
    # accuracy_dir = '/movie_recommendations/results_train/'
    # train_accuracy_dir = '/movie_recommendations/results_train/'

    # epochs = 100
    # lower_lr = 0.0001
    # upper_lr = 0.001
    # prefix = 'audio'
    # accuracy_dir = '/audio/results_easy/'
    # train_accuracy_dir = '/audio/results_train/'

    epochs = 50
    lower_lr = 0.0001
    upper_lr = 0.001
    prefix = 'lenet'

    accuracy_dir = ""
    train_accuracy_dir = ""

    stats_dir = accuracy_dir + 'stats/'
    train_stats_dir = train_accuracy_dir + 'stats/'
    killed_name_list = []
    get_overall_mutation_score(stats_dir, train_stats_dir)
    #print(postprocess_killed_name_list())

