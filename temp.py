import json
import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from data_load_json import PlanningDataset
from utils import *
from utils.args import get_args
from train_mlp import ResMLP, TransformerHead
from utils.env_args import get_environment_shape
from tqdm import tqdm


def main():
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # deploy the specific dataset
    env_dict = get_environment_shape(args.dataset, args.horizon)
    args.action_dim = env_dict['action_dim']
    args.observation_dim = env_dict['observation_dim']
    args.class_dim = env_dict['class_dim']
    args.root = env_dict['root']
    args.json_path_train = env_dict['json_path_train']
    args.json_path_val = env_dict['json_path_val']
    args.json_path_val2 = env_dict['json_path_val2']
    args.n_diffusion_steps = env_dict['n_diffusion_steps']
    args.n_train_steps = env_dict['n_train_steps']
    args.epochs = env_dict['epochs']
    args.lr = env_dict['lr']

    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f'tcp://localhost:{args.dist_port}',
            world_size=args.world_size,
            rank=args.rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(
                args.num_thread_reader / ngpus_per_node)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )

    # create model
    model = TransformerHead(
        input_dim=args.observation_dim, output_dim=args.class_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout)
    # model = head(args.observation_dim, args.class_dim)

    if args.distributed:
        if args.gpu is not None:
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)

    elif args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    checkpoint_ = torch.load(args.ckpt_path,
                             map_location='cuda:{}'.format(args.gpu))
    model.load_state_dict(checkpoint_["model"])

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    model.eval()

    # Initialize an empty list to store the results
    json_ret = []

    # Initialize a counter for correctly classified samples
    correct = 0

    # Iterate over the length of the test dataset
    for i in tqdm(range(len(test_dataset)), desc='inference'):
        # Load the frames, video names, and frame counts from the test dataset
        frames_t, vid_names, frame_cnts = test_dataset[i]

        # Predict the event class using the model
        event_class = model(frames_t)

        # Find the index of the maximum value in the predicted event class tensor
        event_class_id = torch.argmax(event_class)

        # Check if the predicted event class matches the actual task ID
        if event_class_id == vid_names['task_id']:
            correct += 1

        # Add the predicted event class ID to the video names dictionary
        vid_names['event_class'] = event_class_id.item()

        # Initialize a new dictionary to store the current result
        json_current = {}

        # Store the video names dictionary in the current result
        json_current['id'] = vid_names

        # Store the instruction length (frame counts) in the current result
        json_current['instruction_len'] = frame_cnts

        # Append the current result to the list of all results
        json_ret.append(json_current)

    # Define the output filename
    data_name = "output.json"

    # Open the file in write mode
    with open(data_name, 'w') as f:
        # Write the list of results to the JSON file
        json.dump(json_ret, f)

    # Print the accuracy of the model predictions
    print('acc: ', correct / len(test_dataset))


if __name__ == "__main__":
    main()
