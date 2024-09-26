import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from collections import namedtuple


Batch = namedtuple('Batch', 'Observations Actions Class')


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

        if 'crosstask' in args.dataset:
            if is_val:
                cross_task_data_name = args.json_path_val2
            else:
                cross_task_data_name = args.json_path_train.replace(
                    ".json", f"_{args.horizon}.json")

            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                assert 0
        elif args.dataset == 'coin':
            if is_val:
                coin_data_name = args.json_path_val2
            else:
                coin_data_name = args.json_path_train.replace(
                    ".json", f"_{args.horizon}.json")

            if os.path.exists(coin_data_name):
                with open(coin_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(coin_data_name))
            else:
                assert 0
        elif args.dataset == 'NIV':
            if is_val:
                niv_data_name = args.json_path_val2
            else:
                niv_data_name = args.json_path_train.replace(
                    ".json", f"_{args.horizon}.json")
            print(niv_data_name)

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
        """
        Curates a dataset by selecting and concatenating image segments based on specified ranges.

        Args:
        - images (list or np.array): A list or array of images.
        - legal_range (list of tuples): A list of tuples where each tuple contains 
        (start_idx, end_idx, action_label). 
        - start_idx (int): The starting index of the segment.
        - end_idx (int): The ending index of the segment.
        - action_label (int or list): The action label corresponding to the segment.
        - M (int, optional): The number of images to include in each segment. Defaults to 2.

        Returns:
        - images_list (list): A list of concatenated image segments.
        - labels_onehot_list (list): A list of action labels corresponding to the image segments.
        - idx_list (list): A list of starting indices of the segments.
        """

        images_list = []
        labels_onehot_list = []
        idx_list = []

        # Loop through the legal ranges to select and process image segments
        for start_idx, end_idx, action_label in legal_range:
            idx = start_idx
            idx_list.append(idx)

            # Determine the start index for image extraction
            image_start_idx = max(0, idx)

            # Extract M images starting from the determined index
            if image_start_idx + M <= len(images):
                image_start = images[image_start_idx: image_start_idx + M]
            else:
                image_start = images[len(images) - M: len(images)]

            # Concatenate the extracted images into a single image
            image_start_cat = image_start[0]
            for w in range(len(image_start) - 1):
                image_start_cat = np.concatenate(
                    (image_start_cat, image_start[w + 1]), axis=0)

            images_list.append(image_start_cat)
            labels_onehot_list.append(action_label)

        # Ensure the end_idx is at least 2
        end_idx = max(2, end_idx)

        # Extract M images ending at the determined index
        image_end = images[end_idx - 2:end_idx + M - 2]

        # Concatenate the extracted images into a single image
        image_end_cat = image_end[0]
        for w in range(len(image_end) - 1):
            image_end_cat = np.concatenate(
                (image_end_cat, image_end[w + 1]), axis=0)

        images_list.append(image_end_cat)
        return images_list, labels_onehot_list, idx_list

    def sample_single(self, index):
        """
        Samples a single video sequence for training or validation.

        Args:
        - index (int): Index of the video in the dataset.

        Returns:
        - frames (torch.Tensor): Tensor containing frames from the video.
        - labels_tensor (torch.Tensor): Tensor containing labels for each frame.
        - event_class/task_class (torch.Tensor): Tensor containing the class label for the video.
        """
        # Get folder information based on the index
        folder_id = self.vid_names[index]

        # Determine if we are in validation mode
        if self.is_val:
            event_class = folder_id['event_class']
        else:
            task_class = folder_id['task_id']
        
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
        feature_filename = os.path.basename(folder_id['feature'])
        feature_path = os.path.join(full_path, feature_filename)
        # Load video frames based on the dataset type
        if 'crosstask' in self.args.dataset:
            # If the video ID is different from the last one loaded, load new frames
            if folder_id['vid'] != self.last_vid:
                images_ = np.load(feature_path, allow_pickle=True)

                if self.args.dataset == 'crosstask_base':
                    self.images = np.array(images_, dtype=np.float32)
                else:
                    self.images = images_['frames_features']

                self.last_vid = folder_id['vid']
        else:
            # Load video frames for other datasets
            images_ = np.load(feature_path, allow_pickle=True)
            self.images = images_['frames_features']

        # Curate the dataset to get frames and labels within the legal range
        images, labels_matrix, idx_list = self.curate_dataset(
            self.images, folder_id['legal_range'], M=self.M)

        # Convert the images and labels to torch tensors
        frames = torch.tensor(np.array(images))
        labels_tensor = torch.tensor(labels_matrix, dtype=torch.long)

        # Return the appropriate class label based on the mode (validation or training)
        if self.is_val:
            event_class = torch.tensor(event_class, dtype=torch.long)
            return frames, labels_tensor, event_class
        else:
            task_class = torch.tensor(task_class, dtype=torch.long)
            return frames, labels_tensor, task_class

    def __getitem__(self, index):
        if self.is_val:
            frames, labels, event_class = self.sample_single(index)
        else:
            frames, labels, task = self.sample_single(index)
        if self.is_val:
            batch = Batch(frames, labels, event_class)
        else:
            batch = Batch(frames, labels, task)
        return batch

    def __len__(self):
        return len(self.json_data)
