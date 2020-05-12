"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
from __future__ import print_function
from __future__ import absolute_import
import yaml


def yaml_parser(input_data):
    """

    :param input_data:
    :return:
    """

    output = None

    if input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.safe_load(input_data)
        except yaml.YAMLError as exc:
            print(exc)

    return output
