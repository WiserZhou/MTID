import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from collections import namedtuple

Batch = namedtuple('Batch', 'Observations json_id json_len')


class PlanningDataset(Dataset):
    """
    load video and action features from dataset
    """

    def __init__(self,
                 root,
                 args=None,
                 is_val=False,
                 model=None,
                 ):
        self.is_val = is_val
        self.data_root = root
        self.args = args
        self.max_traj_len = args.horizon
        self.vid_names = None
        self.frame_cnts = None
        self.images = None
        self.last_vid = ''
        self.device = torch.device(
            f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")

        if 'crosstask' in args.dataset:
            cross_task_data_name = args.json_path_val.replace(
                    ".json", f"_{args.horizon}.json")
            # print(cross_task_data_name)
            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                assert 0
        elif args.dataset == 'coin':
            coin_data_name = args.json_path_val.replace(
                    ".json", f"_{args.horizon}.json")
           
            if os.path.exists(coin_data_name):
                with open(coin_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(coin_data_name))
            else:
                assert 0
        elif args.dataset == 'NIV':
            niv_data_name = args.json_path_val.replace(
                    ".json", f"_{args.horizon}.json")
           
            if os.path.exists(niv_data_name):
                with open(niv_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(niv_data_name))
            else:
                assert 0
        else:
            raise NotImplementedError(
                'Dataset {} is not implemented'.format(args.dataset))

        self.model = model
        self.prepare_data()
        self.M = 3

    def prepare_data(self):
        vid_names = []
        frame_cnts = []
        for listdata in self.json_data:
            vid_names.append(listdata['id'])
            frame_cnts.append(listdata['instruction_len'])
        self.vid_names = vid_names
        self.frame_cnts = frame_cnts

    def curate_dataset(self, images, legal_range, M=2):
        images_list = []
        labels_onehot_list = []
        idx_list = []
        for start_idx, end_idx, action_label in legal_range:
            idx = start_idx
            idx_list.append(idx)
            image_start_idx = max(0, idx)
            if image_start_idx + M <= len(images):
                image_start = images[image_start_idx: image_start_idx + M]
            else:
                image_start = images[len(images) - M: len(images)]
            image_start_cat = image_start[0]
            for w in range(len(image_start) - 1):
                image_start_cat = np.concatenate(
                    (image_start_cat, image_start[w + 1]), axis=0)

            images_list.append(image_start_cat)
            labels_onehot_list.append(action_label)

        end_idx = max(2, end_idx)
        image_end = images[end_idx - 2:end_idx + M - 2]
        image_end_cat = image_end[0]
        for w in range(len(image_end) - 1):
            image_end_cat = np.concatenate(
                (image_end_cat, image_end[w + 1]), axis=0)
        images_list.append(image_end_cat)

        return images_list, labels_onehot_list, idx_list

    def sample_single(self, index):
        folder_id = self.vid_names[index]
        
          # Define dataset paths
        self.dataset_paths = {
            "crosstask_how": "crosstask/processed_data/",
            "crosstask_base": "crosstask/crosstask_features/",
            "coin": "coin/full_npy/",
            "NIV": "NIV/processed_data/"
        }
        
        self.current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Determine the dataset path
        dataset_path = self.dataset_paths.get(self.args.dataset, "")
        full_path = os.path.join(self.current_dir, "dataset", dataset_path)
        
        # print('---------------------------')
        # print(folder_id['feature'])
        
        feature_filename = os.path.basename(folder_id['feature'])
        
        # print(feature_filename)
        feature_path = os.path.join(full_path, feature_filename)
        
        if 'crosstask' in self.args.dataset:
            if folder_id['vid'] != self.last_vid:
                images_ = np.load(feature_path, allow_pickle=True)
                if self.args.dataset == 'crosstask_base':
                    self.images = images_
                else:
                    self.images = images_['frames_features']
                self.last_vid = folder_id['vid']
        else:
            # print(self.args.dataset)
            images_ = np.load(feature_path, allow_pickle=True)
            self.images = images_['frames_features']
        images, labels_matrix, idx_list = self.curate_dataset(
            self.images, folder_id['legal_range'], M=self.M)
        frames = torch.tensor(np.array(images))
        return frames

    def __getitem__(self, index):

        frames = self.sample_single(index)
        frames_t = torch.zeros(
            2, self.args.observation_dim, device=self.device)
        frames_t[0, :] = frames[0, :].to(self.device)
        frames_t[1, :] = frames[-1, :].to(self.device)
        frames_t = frames_t.view(1, 2, -1)

        batch = Batch(frames_t, self.vid_names[index], self.frame_cnts[index])

        return batch

    def __len__(self):
        return len(self.json_data)
