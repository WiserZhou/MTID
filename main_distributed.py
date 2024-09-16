import glob
import os
import random
import time
from collections import OrderedDict

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import ReduceOp
from utils.env_args import get_environment_shape

import utils
from dataloader.data_load import PlanningDataset
from model.helpers import get_lr_schedule_with_warmup

from utils import *
# from logging import log
from utils.args import get_args
import numpy as np
from model.helpers import Logger
from tqdm import tqdm
from inference import test_inference
from utils.accuracy import parse_fraction_or_float
import wandb


def reduce_tensor(tensor):
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    else:
        return tensor


def main():
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.set_num_threads(20)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

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
        torch.multiprocessing.set_start_method('spawn')
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.base_model != 'base':
        base_model = 'predictor'
    else:
        base_model = 'base'
    env_dict = get_environment_shape(args.dataset, args.horizon,base_model)
    args.action_dim = env_dict['action_dim']
    args.observation_dim = env_dict['observation_dim']
    args.class_dim = env_dict['class_dim']
    args.root = env_dict['root']
    args.json_path_train = env_dict['json_path_train']
    args.json_path_val = env_dict['json_path_val']
    args.json_path_val2 = env_dict['json_path_val2']
    args.n_diffusion_steps = env_dict['n_diffusion_steps']
    args.n_train_steps = env_dict['n_train_steps']
    epoch_env = env_dict['epochs']
    if args.dataset != 'coin':
        args.lr = env_dict['lr']
        
    current_time = time.strftime("%Y%m%d_%H%M%S")
    wandb_name = f"{args.base_model}_{args.name}_{current_time}"

    if args.verbose:
        print(args)

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

    # Data loading code
    train_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=False,
        model=None,
    )
    # Test data loading code
    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )
    from model import diffusion, temporal, temporalPredictor,temporalPreAll,temporalPreStep,temporalPredictorCasual

    # create model
    if args.base_model == 'base':
        temporal_model = temporal.TemporalUnet(args,dim=256,dim_mults=(1, 2, 4), )
    elif args.base_model == 'predictor':
        temporal_model = temporalPredictor.TemporalUnet(args,dim=256,dim_mults=(1, 2, 4), )
    elif args.base_model == 'preAll':
        temporal_model = temporalPreAll.TemporalUnet(args,dim=256,dim_mults=(1, 2, 4), )
    elif args.base_model == 'preStep':
        temporal_model = temporalPreStep.TemporalUnet(args,dim=256,dim_mults=(1, 2, 4), )
    elif args.base_model == 'preCas':
        temporal_model = temporalPredictorCasual.TemporalUnet(args,dim=256,dim_mults=(1, 2, 4),)
    else:
        RuntimeError('unvalid base model!')
    
        
    if args.base_model != 'base':
        args.base_model = 'predictor'

    diffusion_model = diffusion.GaussianDiffusion(
        args, temporal_model)

    model = utils.Trainer(args, diffusion_model, train_loader)

    if args.distributed:
        if args.gpu is not None:
            model.model.cuda(args.gpu)
            model.ema_model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, device_ids=[args.gpu], find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(
                model.ema_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.model.cuda()
            model.ema_model.cuda()
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(model.ema_model,
                                                                        find_unused_parameters=True)

    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.ema_model = torch.nn.DataParallel(model.ema_model).cuda()
        
    scale1 = parse_fraction_or_float(args.scale1)
    scale2 = parse_fraction_or_float(args.scale2)
    
    scheduler = get_lr_schedule_with_warmup(
        model.optimizer, int(args.n_train_steps * epoch_env),
        args.dataset,args.base_model,
        args.schedule
        # scale1=scale1, scale2=scale2,
        )

    checkpoint_dir = os.path.join(os.path.dirname(
        __file__), 'checkpoint', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not (os.path.isdir(checkpoint_dir)) and args.rank == 0:
        os.mkdir(checkpoint_dir)

    if args.resume:
        if args.resume_path == 'None':
            checkpoint_path = get_last_checkpoint(checkpoint_dir,args.name)
        else:
            checkpoint_path = args.resume_path
        
        if args.epochs != None:
            epoch_env = args.epochs
            
        print('load checkpoint path:',checkpoint_path)
        if checkpoint_path:
            print("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(
                checkpoint_path, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"])
            model.ema_model.load_state_dict(checkpoint["ema"])
            model.optimizer.load_state_dict(checkpoint["optimizer"])
            model.step = checkpoint["step"]
            scheduler.load_state_dict(checkpoint["scheduler"])
            
    if args.rank == 0:
        wandb.init(project=f"MTID_{args.dataset}_T{args.horizon}", name=wandb_name, config=args)
        wandb.watch(model.model)


    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size
    print(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ), args
    )

    max_eva = 0
    max_acc = 0
    old_max_epoch = 0
    save_max = os.path.join(os.path.dirname(__file__), 'save_max')

    ckpt_max_path = ''

    max_train_acc = 0
    max_train_epoch = 0
    max_test_acc = 0
    max_test_epoch = 0

    

    # Main training loop across epochs
    for epoch in tqdm(range(args.start_epoch, epoch_env), desc='total train'):

        # If distributed training is enabled, set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if (epoch + 1) % 5 == 0:  # calculate on training set
            losses, acc_top1, acc_top5, trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, \
                acc_a0, acc_aT = model.train(
                    args.n_train_steps, True, args, scheduler)

            # max_train_acc = max(max_train_acc, trajectory_success_rate_meter)
            if trajectory_success_rate_meter > max_train_acc:
                max_train_acc = trajectory_success_rate_meter
                max_train_epoch = epoch + 1

            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
            acc_top5_reduced = reduce_tensor(acc_top5.cuda()).item()
            trajectory_success_rate_meter_reduced = reduce_tensor(
                trajectory_success_rate_meter.cuda()).item()
            MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
            MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
            acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
            acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

            if args.rank == 0:
                logs = OrderedDict()
                logs['Train/EpochLoss'] = losses_reduced
                logs['Train/EpochAcc@1'] = acc_top1_reduced
                logs['Train/EpochAcc@5'] = acc_top5_reduced
                logs['Train/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced
                logs['Train/MIoU1'] = MIoU1_meter_reduced
                logs['Train/MIoU2'] = MIoU2_meter_reduced
                logs['Train/acc_a0'] = acc_a0_reduced
                logs['Train/acc_aT'] = acc_aT_reduced
                wandb.log(logs, step=epoch + 1)
        else:
            losses = model.train(args.n_train_steps, False,
                                 args, scheduler).cuda()
            losses_reduced = reduce_tensor(losses).item()
            if args.rank == 0:
                print()
                for p in model.optimizer.param_groups:
                    print('lrs:' + str(p['lr']))
                print('---------------------------------')

                logs = OrderedDict()
                logs['Train/lrs'] = p['lr']
                logs['Train/EpochLoss'] = losses_reduced
                wandb.log(logs, step=epoch + 1)

        # Evaluate the model every epochs if evaluation is enabled
        if args.evaluate:
            losses, acc_top1, acc_top5, \
                trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, \
                acc_a0, acc_aT = validate(test_loader, model.ema_model, args)

            # max_test_acc = max(max_test_acc, trajectory_success_rate_meter)
            if trajectory_success_rate_meter > max_test_acc:
                max_test_acc = trajectory_success_rate_meter
                max_test_epoch = epoch + 1

            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
            acc_top5_reduced = reduce_tensor(acc_top5.cuda()).item()
            trajectory_success_rate_meter_reduced = reduce_tensor(
                trajectory_success_rate_meter.cuda()).item()
            MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
            MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
            acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
            acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

            # If the current process is the main process (rank 0), log the validation metrics
            if args.rank == 0:
                logs = OrderedDict()
                logs['Val/EpochLoss'] = losses_reduced
                logs['Val/EpochAcc@1'] = acc_top1_reduced
                logs['Val/EpochAcc@5'] = acc_top5_reduced
                logs['Val/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced
                logs['Val/MIoU1'] = MIoU1_meter_reduced
                logs['Val/MIoU2'] = MIoU2_meter_reduced
                logs['Val/acc_a0'] = acc_a0_reduced
                logs['Val/acc_aT'] = acc_aT_reduced
                wandb.log(logs, step=epoch + 1)
                
                print(trajectory_success_rate_meter_reduced,acc_top1_reduced,MIoU2_meter_reduced,MIoU1_meter_reduced, max_eva)

            # Save checkpoint if the new trajectory success rate is better
            if trajectory_success_rate_meter_reduced > max_eva and acc_top1_reduced >= max_acc:
                # if not (trajectory_success_rate_meter_reduced == max_eva and acc_top1_reduced < max_acc):
                ckpt_max_path = save_checkpoint_max(args.name,
                                                    {
                                                        "epoch": epoch + 1,
                                                        "model": model.model.state_dict(),
                                                        "ema": model.ema_model.state_dict(),
                                                        "optimizer": model.optimizer.state_dict(),
                                                        "step": model.step,
                                                        "scheduler": scheduler.state_dict(),
                                                    }, save_max, old_max_epoch, epoch + 1, args.rank
                                                    )
                max_eva = trajectory_success_rate_meter_reduced
                max_acc = acc_top1_reduced
                old_max_epoch = epoch + 1

        # Save checkpoint periodically based on the save frequency
        if (epoch + 1) % args.save_freq == 0:
            if args.rank == 0:
                save_checkpoint(args.name,
                                {
                                    "epoch": epoch + 1,
                                    "model": model.model.state_dict(),
                                    "ema": model.ema_model.state_dict(),
                                    "optimizer": model.optimizer.state_dict(),
                                    "step": model.step,
                                    "scheduler": scheduler.state_dict(),
                                }, checkpoint_dir, epoch + 1
                                )
                
    print(f'max_train_acc:{max_train_acc} max_train_epoch:{max_train_epoch}')
    print(f'max_test_acc:{max_test_acc} max_test_epoch:{max_test_epoch}')

    # add inference
    if ckpt_max_path == '':
        ckpt_max_path = get_last_checkpoint(args.checkpoint_max_root,args.name)
        
    ckpt_max_path = os.path.join(args.checkpoint_max_root, ckpt_max_path)
    
    if ckpt_max_path:
        print("=> loading checkpoint '{}'".format(ckpt_max_path), args)
        checkpoint = torch.load(
            ckpt_max_path, map_location='cuda:{}'.format(args.gpu))
        args.start_epoch = checkpoint["epoch"]
        model.model.load_state_dict(checkpoint["model"], strict=True)
        model.ema_model.load_state_dict(checkpoint["ema"], strict=True)
        model.step = checkpoint["step"]

    time_start = time.time()
    acc_top1_reduced_sum = []
    trajectory_success_rate_meter_reduced_sum = []
    MIoU1_meter_reduced_sum = []
    MIoU2_meter_reduced_sum = []
    acc_a0_reduced_sum = []
    acc_aT_reduced_sum = []
    test_times = 1

    for epoch in range(0, test_times):
        # tmp = epoch
        # random.seed(tmp)
        # np.random.seed(tmp)
        # torch.manual_seed(tmp)
        # torch.cuda.manual_seed_all(tmp)

        acc_top1, trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, acc_a0, acc_aT = test_inference(
            test_loader, model.ema_model, args)

        acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
        trajectory_success_rate_meter_reduced = reduce_tensor(
            trajectory_success_rate_meter.cuda()).item()
        MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
        MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
        acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
        acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

        acc_top1_reduced_sum.append(acc_top1_reduced)
        trajectory_success_rate_meter_reduced_sum.append(
            trajectory_success_rate_meter_reduced)
        MIoU1_meter_reduced_sum.append(MIoU1_meter_reduced)
        MIoU2_meter_reduced_sum.append(MIoU2_meter_reduced)
        acc_a0_reduced_sum.append(acc_a0_reduced)
        acc_aT_reduced_sum.append(acc_aT_reduced)

    if args.rank == 0:
        time_end = time.time()

        timeP = time_end - time_start
        EpochAcc1 = sum(acc_top1_reduced_sum) / test_times
        Traj_Success_Rate = sum(
            trajectory_success_rate_meter_reduced_sum) / test_times
        MIoU2 = sum(MIoU2_meter_reduced_sum) / test_times

        print('time: ', timeP)
        print('-----------------Mean&Var-----------------------')
        print('Val/EpochAcc@1', EpochAcc1)
        print('Val/Traj_Success_Rate', Traj_Success_Rate)
        print('Val/MIoU1', sum(MIoU1_meter_reduced_sum) /
              test_times, np.var(MIoU1_meter_reduced_sum))
        print('Val/MIoU2', MIoU2)
        print('Val/acc_a0', sum(acc_a0_reduced_sum) /
              test_times, np.var(acc_a0_reduced_sum))
        print('Val/acc_aT', sum(acc_aT_reduced_sum) /
              test_times, np.var(acc_aT_reduced_sum))
        
        wandb.log({
            'Final/EpochAcc@1': EpochAcc1,
            'Final/Traj_Success_Rate': Traj_Success_Rate,
            'Final/MIoU1': sum(MIoU1_meter_reduced_sum) / test_times,
            'Final/MIoU2': MIoU2,
            'Final/acc_a0': sum(acc_a0_reduced_sum) / test_times,
            'Final/acc_aT': sum(acc_aT_reduced_sum) / test_times,
        })
        wandb.finish()

        # print experiment results
        print(f"{args.name} {args.dataset} {args.horizon} {old_max_epoch} {Traj_Success_Rate} {EpochAcc1} {MIoU2}")


# def log(output, args):
#     with open(os.path.join(os.path.dirname(__file__), 'log', args.checkpoint_dir + '.txt'), "a") as f:
#         f.write(output + '\n')


def save_checkpoint(name, state, checkpoint_dir, epoch, n_ckpt=3):
    torch.save(state, os.path.join(checkpoint_dir,
               "epoch_{}_{:0>4d}.pth.tar".format(name, epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(
            checkpoint_dir, "epoch_{}_{:0>4d}.pth.tar".format(name, epoch - n_ckpt))
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def save_checkpoint_max(name, state, checkpoint_dir, old_epoch, epoch, rank):

    cktp_name = "epoch_{}_{:0>4d}_{}.pth.tar".format(name, epoch, rank)
    torch.save(state, os.path.join(checkpoint_dir,
               "epoch_{}_{:0>4d}_{}.pth.tar".format(name, epoch, rank)))
    if old_epoch > 0:
        oldest_ckpt = os.path.join(
            checkpoint_dir, "epoch_{}_{:0>4d}_{}.pth.tar".format(name, old_epoch, rank))
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)
    return cktp_name


def get_last_checkpoint(checkpoint_dir,name):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, f'epoch_{name}_*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''


if __name__ == "__main__":
    main()
