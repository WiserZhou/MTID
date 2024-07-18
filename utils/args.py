import argparse


def get_args(description='whl'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint_root',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/checkpoint',
                        help='checkpoint dir root')
    parser.add_argument('--log_root',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/log/log',
                        help='log dir root')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='whl',
                        help='checkpoint model folder')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='opt algorithm')
    parser.add_argument('--num_thread_reader',
                        type=int,
                        default=8,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='batch size')
    parser.add_argument('--batch_size_val',
                        type=int,
                        default=256,
                        help='batch size eval')
    parser.add_argument('--pretrain_cnn_path',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--momemtum',
                        type=float,
                        default=0.9,
                        help='SGD momemtum')
    parser.add_argument('--log_freq',
                        type=int,
                        default=500,
                        help='how many steps do we log once')
    parser.add_argument('--save_freq',
                        type=int,
                        default=1,
                        help='how many epochs do we save once')
    parser.add_argument('--gradient_accumulate_every',
                        type=int,
                        default=1,
                        help='accumulation_steps')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.995,
                        help='')
    parser.add_argument('--step_start_ema',
                        type=int,
                        default=400,
                        help='')
    parser.add_argument('--update_ema_every',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--crop_only',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--centercrop',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--random_flip',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--fps',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--cudnn_benchmark',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--dataset',
                        type=str,
                        default='crosstask',
                        help='dataset')
    parser.add_argument('--action_dim',
                        type=int,
                        default=105,
                        help='')
    parser.add_argument('--observation_dim',
                        type=int,
                        default=1536,
                        help='')
    parser.add_argument('--class_dim',
                        type=int,
                        default=18,
                        help='')
    parser.add_argument('--n_diffusion_steps',
                        type=int,
                        default=200,
                        help='')
    parser.add_argument('--n_train_steps',
                        type=int,
                        default=200,
                        help='training_steps_per_epoch')
    parser.add_argument('--root',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/dataset/crosstask',
                        help='root path of dataset crosstask')
    parser.add_argument('--json_path_train',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/train_list.json',
                        help='path of the generated json file for train')
    parser.add_argument('--json_path_val',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/test_list.json',
                        help='path of the generated json file for val train mlp')
    parser.add_argument('--json_path_val2',
                        type=str,
                        default='/home/zhouyufan/Projects/PDPP/dataset/crosstask/crosstask_release/output.json',
                        help='path of the generated json file for val train model')

    parser.add_argument('--epochs', default=190, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('-e', '--evaluate', default=True, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pin_memory', default=True, dest='pin_memory', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default='dist-file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=217, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=6, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--name', default='default', type=str,
                        help='note the specific log and checkpoint')
    parser.add_argument('--loss_kind', default='Weighted_MSE', type=str,
                        help='Weighted_MSE: only 10 on both side ; Weighted_Gradient_MSE: gradient change')
    parser.add_argument('--ckpt_path', default='', type=str,
                        help='checkpoint path for max')
    parser.add_argument('--layer_num', default=3, type=int,
                        help='number of layers in block')
    parser.add_argument('--dist_port', default=21712, type=int,
                        help='port used to set up distributed training')
    parser.add_argument('--horizon',
                        type=int,
                        default=3,
                        help='')
    parser.add_argument('--weight', default=10, type=float,
                        help='weight of the loss function')
    args = parser.parse_args()
    return args
