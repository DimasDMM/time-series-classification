import tensorflow as tf
import os
import json

class ModelLoader:
    def __init__(self, artifacts_path):
        self.__artifacts_path = artifacts_path
        self.__init_artifacts_path()
    
    def __init_artifacts_path(self):
        if not os.path.exists(self.__artifacts_path):
            os.mkdir(self.__artifacts_path)

    def __get_model_path(self, model_name):
        return '{}/{}'.format(self.__artifacts_path, model_name)
    
    def create_model_dir(self, model_name):
        model_path = self.__get_model_path(model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

    def save_tf_model(self, model_name, model):
        model_path = self.__get_model_path(model_name)
        tf.saved_model.save(model, model_path)

    def save_evaluation_results(self, model_name, results):
        model_path = self.__get_model_path(model_name)
        json_file = '{}/evaluation_results.json'.format(model_path)
        with open(json_file, 'w') as fp:
            json.dump(results, fp)

    def load_evaluation_results(self, model_name):
        model_path = self.__get_model_path(model_name)
        json_file = '{}/evaluation_results.json'.format(model_path)
        with open(json_file, 'r') as fp:
            return json.load(fp)

    def save_training_parameters(self, model_name, params):
        model_path = self.__get_model_path(model_name)
        json_file = '{}/training_params.json'.format(model_path)
        with open(json_file, 'w') as fp:
            json.dump(params, fp)

    def load_training_parameters(self, model_name):
        model_path = self.__get_model_path(model_name)
        json_file = '{}/training_params.json'.format(model_path)
        with open(json_file, 'r') as fp:
            return json.load(fp)
