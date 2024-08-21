import os


def get_current_file_absolute_path():
    current_file_path = os.path.abspath(__file__)
    path_parts = current_file_path.split('/')
    before_utils = '/'.join(path_parts[:-2])
    return before_utils


class EnvironmentInfo:
    def __init__(self, name, observation_dim, action_dim, class_dim, json_path_train='',
                 json_path_val='', json_path_val3='', json_path_val4='', json_path_val5='',
                 json_path_val6='', n_diffusion_steps=200, n_train_steps=200, epochs=600, lr=5e-4):
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.root = get_current_file_absolute_path() + '/dataset/' + \
            name.split('_')[0]

        self.json_path_train = json_path_train
        self.json_path_val = json_path_val
        self.json_path_val3 = json_path_val3
        self.json_path_val4 = json_path_val4
        self.json_path_val5 = json_path_val5
        self.json_path_val6 = json_path_val6
        self.n_diffusion_steps = n_diffusion_steps
        self.n_train_steps = n_train_steps
        self.epochs = epochs
        self.lr = lr


# Define the environments based on the provided data
environments = [
    EnvironmentInfo("crosstask_how",  1536, 105, 18,
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/train_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/test_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/output3.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/output4.json',
                    '',
                    '',
                    200,
                    200,
                    120,
                    5e-4),
    EnvironmentInfo("crosstask_base",  9600, 105, 18,
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/train_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/test_list.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/base/output3.json',
                    '',
                    '',
                    '',
                    200,
                    200,
                    60,
                    8e-4),
    EnvironmentInfo("coin", 1536, 778, 180,
                    '/home/zhouyufan/Projects/PDPP/dataset/coin/coin_train_70.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/coin/coin_test_30.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/coin/output3.json',
                    '',
                    '',
                    '',
                    200,
                    200,
                    800,
                    1e-5),
    EnvironmentInfo("NIV", 1536, 48, 5,
                    '/home/zhouyufan/Projects/PDPP/dataset/NIV/train70.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/NIV/test30.json',
                    '/home/zhouyufan/Projects/PDPP/dataset/NIV/output3.json',
                    '',
                    '',
                    '',
                    50,
                    50,
                    130,
                    3e-4)
]


def get_environment_shape(name, horizon):
    for env in environments:
        if env.name == name:
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
