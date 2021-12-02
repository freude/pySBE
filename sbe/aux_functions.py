"""
The module contains a set of auxiliary functions facilitating the tight-binding computations
"""
from __future__ import print_function
from __future__ import absolute_import
import yaml
import importlib


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


def import_check(module_name='sys'):
    def check(func):
        try:
            module = importlib.import_module(module_name)
            # exec("import "+module_name)
            return func
        except ModuleNotFoundError:
            pass
    return check