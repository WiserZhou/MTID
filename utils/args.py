import argparse


def get_args(description='whl'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint_root',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/checkpoint',
                        help='checkpoint dir root')
    parser.add_argument('--checkpoint_max_root',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/save_max',
                        help='checkpoint max dir root')
    parser.add_argument('--log_root',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/log/log',
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
    parser.add_argument('--momemtum',
                        type=float,
                        default=0.9,
                        help='SGD momemtum')
    parser.add_argument('--save_freq',
                        type=int,
                        default=1,
                        help='how many epochs do we save once')
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

    # kind of dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='crosstask_how',
                        help='dataset:crosstask_how,crosstask_base,coin,NIV')
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
    parser.add_argument('--root',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/dataset/crosstask',
                        help='root path of dataset crosstask')
    parser.add_argument('--json_path_train',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/dataset/crosstask/crosstask_release/train_list.json',
                        help='path of the generated json file for train')
    parser.add_argument('--json_path_val',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/dataset/crosstask/crosstask_release/test_list.json',
                        help='path of the generated json file for val train mlp')
    parser.add_argument('--json_path_val2',
                        type=str,
                        default='/data/zhaobo/zhouyufan/MTID/dataset/crosstask/crosstask_release/output.json',
                        help='path of the generated json file for val train model')
    ########################################################################################

    parser.add_argument('--n_train_steps',
                        type=int,
                        default=200,
                        help='training_steps_per_epoch')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--resume_path', default='None',type=str,
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
    parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--name', default='default', type=str,
                        help='note the specific log and checkpoint')

    parser.add_argument('--ckpt_path', default='', type=str,
                        help='checkpoint path for max')
    parser.add_argument('--dist_port', default=21712, type=int,
                        help='port used to set up distributed training')
    parser.add_argument('--log_freq',
                        type=int,
                        default=500,
                        help='how many steps do we log once')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # parameters that need to be modified
    parser.add_argument('--seed', default=217, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--clip_denoised', default=True,
                        action='store_true', help='')
    parser.add_argument('--ddim_discr_method',
                        default='uniform', type=str, help='quad or uniform')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.995,
                        help='')
    parser.add_argument('--gradient_accumulate_every',
                        type=int,
                        default=1,
                        help='accumulation_steps')
    parser.add_argument('--step_start_ema',
                        type=int,
                        default=400,
                        help='')
    parser.add_argument('--update_ema_every',
                        type=int,
                        default=10,
                        help='')


    parser.add_argument('--base_model', type=str,
                        default='base', help='predictor')
    parser.add_argument('--classfier_model',default='transformer',
                        type=str,help='classfier model to use')
    parser.add_argument('--n_diffusion_steps',
                        type=int,
                        default=200,
                        help='')
    # train_mlp
    # d_model: 必须大于 0。
    # nhead: 必须大于 0，并且能够整除 d_model。这是因为每个注意力头的维度是 d_model / nhead，所以 d_model 必须能够被 nhead 整除。
    # dim_feedforward: 必须大于 0。通常情况下，dim_feedforward 的值会比 d_model 大。
    # dropout: 必须在 0 到 1 之间。
    #  d_model=dim_feedforward,  # 这里 `d_model` 和 `dim_feedforward` 都设置为同一个值
    #         nhead=num_heads,          # 注意力头的数量
    #         dim_feedforward=dim_feedforward,  # 前馈网络内部的维度
    #         dropout=dropout           # Dropout 概率
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=1024,help='coin:2048')
    parser.add_argument('--dropout', type=float, default=0.4,help='coin:0.7,others:0.4')

    parser.add_argument('--horizon',
                        type=int,
                        default=3,
                        help='')
    parser.add_argument('--epochs', default=None, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--if_jump',default=1,
                        type=int, help='whether to use DDIM to inference')
    
    parser.add_argument('--loss_type', default='Weighted_Gradient_MSE', type=str,
                    help='Weighted_Gradient_MSE: gradient change ; Sequence_CE: CE and order loss;')
    parser.add_argument('--kind',default=0,type=int)
    parser.add_argument('--l_order', default=1000.0, type=float, help='ratio of lambda_order')
    parser.add_argument('--l_pos', default=1.0, type=float, help='ratio of lambda_pos')
    parser.add_argument('--l_perm', default=1.0, type=float, help='ratio of lambda_perm')
    parser.add_argument("--ifMask", type=bool,default=True, help="whether use mask")
    parser.add_argument('--scale1',type=str,default='1/6')
    parser.add_argument('--scale2',type=str,default='1/4')
    parser.add_argument('--schedule',type=str,default='not')
    parser.add_argument('--model_dim',type=int,default=256,help='model dimension')
    parser.add_argument('--module_kind',type=str,default='all',help='ablation for module design')
    parser.add_argument('--encoder_kind',type=str,default='conv')
    parser.add_argument('--mask_loss',type=str,default='none',help='1')
    parser.add_argument('--weight', default=6, type=float,
                        help='weight of the loss function')
    parser.add_argument('--mask_iteration', type=str, default='none', help='add')
    parser.add_argument('--transformer_num', type=int,
                        default=5, help='layer nums for transformer blocks,NIV:2,coin:7')
    parser.add_argument('--ie_num', type=int, default=1,
                        help='image encoder convolution layer num,NIV:2')
    parser.add_argument('--interpolation_init',type=int,default=0,help='interpolation init schema')
    parser.add_argument('--interpolation_usage',type=int,default=0,help='interpolation usage schema')
    parser.add_argument('--mask_scale',type=float,default=1.1,help='scale of mask loss')
    args = parser.parse_args()
    return args

# --loss_type=Weighted_MSE --mask_loss=2 --weight=1
# --loss_type=Weighted_MSE --mask_loss=2 --weight=6
# --loss_type=Weighted_Gradient_MSE --mask_loss=2 --weight=6
# --loss_type=Weighted_MSE --mask_loss=1 --weight=1
# --loss_type=Weighted_MSE --mask_loss=1 --weight=6
# --loss_type=Weighted_Gradient_MSE --mask_loss=1 --weight=6