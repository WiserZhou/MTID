import os


def get_current_file_absolute_path():
    current_file_path = os.path.abspath(__file__)
    path_parts = current_file_path.split('/')
    before_utils = '/'.join(path_parts[:-2])
    return before_utils


class EnvironmentInfo:
    def __init__(self, name, observation_dim, action_dim, class_dim, json_path_train='', json_path_val='', json_path_val2=''):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.root = get_current_file_absolute_path() + '/dataset/' + \
            name.split('_')[0]

        self.json_path_train = json_path_train
        self.json_path_val = json_path_val
        self.json_path_val2 = json_path_val2


# Define the environments based on the provided data
environments = [
    EnvironmentInfo("crosstask_how",  1536, 105, 18,
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/train_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/test_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/output.json'),
    EnvironmentInfo("crosstask_base",  9600, 105, 18,
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/train_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/test_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/output.json'),
    EnvironmentInfo("coin", 1536, 778, 180),
    EnvironmentInfo("NIV", 1536, 48, 5)
]


def get_environment_shape(name):
    for env in environments:
        if env.name == name:
            return {
                'observation_dim': env.observation_dim,
                'action_dim': env.action_dim,
                'class_dim': env.class_dim,
                'root': env.root,
                'json_path_train': env.json_path_train,
                'json_path_val': env.json_path_val,
                'json_path_val2': env.json_path_val2
            }
    return None


# # Example usage
# if __name__ == "__main__":
#     # Specify the environment name you want to get information about
#     environment_name = "CrossTask"
#     shape_info = get_environment_shape(environment_name)

#     if shape_info is not None:
#         print(f"Shape information for {environment_name}:")
#         print(shape_info)
#     else:
#         print(f"No environment found with name: {environment_name}")
