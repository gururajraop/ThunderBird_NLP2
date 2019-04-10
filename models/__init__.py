"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 1: Lexical Alignment
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import importlib
from models.Base_model import BaseModel


def find_model_using_name(model_name):
    """
    Import the module "models/[model_name]_model.py".
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("Missing file ", model_name, "_model.py in the models/ directory")
        exit(0)

    return model


def create_model(opt):
    """
    Create the model based on the given options

    :param opt:
    :return:
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
