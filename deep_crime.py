"""Mutate all the things - Main file of the mutation tool

Available functions:
mutate()
    mutates the unmutated

"""

import os
import argparse
import mutate as m
from utils.logger_setup import setup_logger

def mutate():

    """Convert a file size to human-readable form.

    Keyword arguments:
    size -- file size in bytes
    a_kilobyte_is_1024_bytes -- if True (default), use multiples of 1024
                                if False, use multiples of 1000

    Returns: string
    """
    #TODO: do we really need them global?
    global problem_type
    global model_path
    global runs_number

    parser = argparse.ArgumentParser()

    parser.add_argument("--problem_type", "-problem_type", choices=['C', 'R'],
                        type=str, help="C (classification) or R (regression)")

    parser.add_argument("--model_path", "-model_path",
                        type=str, help="path to model")

    parser.add_argument("--runs_number", "-runs_number",
                        type=int, help="number or runs")

    args = parser.parse_args()

    problem_type = args.problem_type

    #TODO: Check if path is valid
    model_path = args.model_path
    model_path = 'test_models/train_self_driving_car.py'
    #runs_number = args.runs_number
    runs_number = 5

    # print('abc', problem_type)
    #TODO: write user defined parrams to a file?
    m.mutate_model(model_path, runs_number)

if __name__ == '__main__':
    logger = setup_logger(__name__)

    logger.info('DeepCrime started')
    mutate()
    logger.info('DeepCrime finished')
