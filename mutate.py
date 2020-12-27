import utils.execute_mutants as exc
from utils.mutation_utils import *
from utils.logger_setup import setup_logger


def mutate_model(file_path, runs_number):
    # TODO: runs_number
    """ Mutate the model

    Keyword arguments:
    file_path -- path to the py file with model

    Returns: ...
    """

    logger = setup_logger(__name__)

    # TODO: parse the file_path, extract the file name, and generate the next 3 var based on it
    model_name = "udacity"#test_mlp_af
    props.model_name = model_name

    # Prepare Model to be muttated
    # TODO: remove this
    mutation_types = ['D', 'H']

    save_path_prepared = const.save_paths["prepared"] + model_name + "_saved.py"
    save_path_trained = "../" + const.save_paths["trained"] + model_name + "_trained.h5"
    # path_trained = ["../" + const.save_paths["trained"], model_name, "_trained.h5"]

    prepare_model(file_path, save_path_prepared, save_path_trained, mutation_types)

    # TODO: get a list of mutations to be applied
    # TODO: go through a list and apply the mutations
    # TODO: make a list of codes for the mutations

    # list of mutations operators to be applied. List can be found in utils.constants

    mutations = ["delete_training_data", "change_label", "add_noise", "unbalance_train_data", "make_output_classes_overlap"]

    mutants_path = const.save_paths["mutated"] + model_name + "/"
    results_path = mutants_path + "results/"

    if not os.path.exists(mutants_path):
        try:
            os.makedirs(mutants_path)
            os.makedirs(results_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))

    for mutation in mutations:
        logger.info("Starting mutation %s", mutation)
        save_path_mutated = mutants_path + model_name + "_" + mutation + "_mutated"

        try:
            mutationClass = create_mutation(mutation)

            mutationClass.mutate(save_path_prepared, save_path_mutated)
        except LookupError as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Was not able to create a class for mutation %s: " + str(e), mutation)
        except Exception as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Unable to apply the mutation for mutation %s: " + str(e), mutation)


        logger.info("Finished mutation %s", mutation)


    exc.execute_original_model(file_path, results_path)

    exc.execute_mutants(mutants_path, mutations)


