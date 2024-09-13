import os
import json

def get_current_file_absolute_path():
    current_file_path = os.path.abspath(__file__)
    path_parts = current_file_path.split('/')
    before_utils = '/'.join(path_parts[:-2])
    return before_utils

class EnvironmentInfo:
    def __init__(self, config, base_path):
        self.__dict__.update(config)
        self.root = os.path.join(base_path, self.name.split('_')[0])
        
        # Update json paths with base_path
        for key in ['json_path_train', 'json_path_val', 'json_path_val3', 'json_path_val4', 'json_path_val5', 'json_path_val6']:
            if hasattr(self, key):
                setattr(self, key, os.path.join(base_path, getattr(self, key)))

# Load environments from JSON file
json_path = os.path.join(os.path.dirname(__file__), 'environments_config.json')
with open(json_path, 'r') as f:
    config = json.load(f)

BASE_PATH = config['base_path']
environments = [EnvironmentInfo(env_config, BASE_PATH) for env_config in config['environments']]

def get_environment_shape(name, horizon, model):
    for env in environments:
        if env.name == name + '_' + str(model):
            json_path_val_key = f'json_path_val{horizon}'
            return {
                'observation_dim': env.observation_dim,
                'action_dim': env.action_dim,
                'class_dim': env.class_dim,
                'root': env.root,
                'json_path_train': env.json_path_train,
                'json_path_val': env.json_path_val,
                'json_path_val2': getattr(env, json_path_val_key, None),
                'n_diffusion_steps': env.n_diffusion_steps,
                'n_train_steps': env.n_train_steps,
                'epochs': env.epochs,
                'lr': env.lr
            }
    return None