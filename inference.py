import os
import random
import time
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import utils

from torch.distributed import ReduceOp
from dataloader.data_load import PlanningDataset
from model import diffusion, temporal, temporalPredictor
from utils import *
from utils.args import get_args



def print_and_size(tensor):
    # 先检查 tensor 是否是 torch.Tensor 类型
    if isinstance(tensor, torch.Tensor):
        print(tensor.size())
        print(tensor)
    else:
        # 如果不是 torch.Tensor，则打印 tensor 的类型和内容
        print(f"Type: {type(tensor)}, Content: {tensor}")


def accuracy2(output, target, topk=(1,), max_traj_len=0):
    # output torch.Size([768, 105])
    # target torch.Size([768])
    with torch.no_grad():
        # 获取最大的 top-k 值
        # print_and_size(topk)  # Type: <class 'tuple'>, Content: (1,)
        maxk = max(topk)

        # print_and_size(maxk)  # Type: <class 'int'>, Content: 1

        # 获取批次大小
        batch_size = target.size(0)  # 768

        # print_and_size(batch_size)  # Type: <class 'int'>, Content: 768

        # 获取预测值中前 maxk 个最大值的索引
        _, pred = output.topk(maxk, 1, True, True)

        # print_and_size(pred)  # torch.Size([768, 1])

        # 转置 pred，使其形状变为 (1, 768)
        pred = pred.t()

        # print_and_size(pred)  # torch.Size([1, 768])

        # pred = language_guide(target,max_traj_len)
        # pred = pred.unsqueeze(0)
        # 将目标值扩展为与 pred 相同的形状，并比较预测值和目标值是否相等
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print_and_size(correct)  # torch.Size([1, 768])

        # 将其重塑为 (batch_size, max_traj_len)
        correct_a = correct[:1].view(-1, max_traj_len)

        # print_and_size(correct_a)  # torch.Size([256, 3])
        correct_all = []
        for i in range(correct_a.shape[1]):
            correct_all.append(correct_a[:,i].reshape(-1).float().mean().mul_(100.0))

        # 计算第一个时间步的平均准确率
        # correct_a0 = correct_a[:, 0].reshape(-1).float().mean().mul_(100.0)
        # # print_and_size(correct_a0)  # torch.Size([])
        # # 计算最后一个时间步的平均准确率
        # correct_aT = correct_a[:, -1].reshape(-1).float().mean().mul_(100.0)
        # # print_and_size(correct_aT)  # torch.Size([])
        # 初始化结果列表
        res = []

        # 对于每个 top-k 值，计算平均准确率
        for k in topk:
            # 重塑 correct 并计算前 k 个预测正确的样本总数
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

            # 将正确预测的样本数转换为百分比，并添加到结果列表中
            res.append(correct_k.mul_(100.0 / batch_size))

        # 只保留第一个时间步的正确预测结果
        correct_1 = correct[:1]

        # 计算整个轨迹的成功率
        trajectory_success = torch.all(correct_1.view(
            correct_1.shape[1] // max_traj_len, -1), dim=1)

        # 计算轨迹成功率的百分比
        trajectory_success_rate = trajectory_success.sum() * 100.0 / \
            trajectory_success.shape[0]

        # 计算 MIoU1
        # 获取预测值中前 1 个最大值的索引
        _, pred_token = output.topk(1, 1, True, True)

        # 将预测值和目标值按照 max_traj_len 重塑
        pred_inst = pred_token.view(correct_1.shape[1], -1)
        target_inst = target.view(correct_1.shape[1], -1)

        # 将预测值和目标值转换为集合
        pred_inst_set = set(tuple(pred_inst[i].tolist())
                            for i in range(pred_inst.shape[0]))
        target_inst_set = set(
            tuple(target_inst[i].tolist()) for i in range(target_inst.shape[0]))

        # 计算 MIoU1
        MIoU1 = 100.0 * len(pred_inst_set.intersection(target_inst_set)
                            ) / len(pred_inst_set.union(target_inst_set))

        # 重新计算批次大小，考虑到 max_traj_len
        batch_size = batch_size // max_traj_len

        # 将预测值和目标值重塑为 (batch_size, max_traj_len)
        pred_inst = pred_token.view(batch_size, -1)  # [bs, T]
        target_inst = target.view(batch_size, -1)  # [bs, T]

        # 初始化 MIoU2 的总和
        MIoU_sum = 0

        # 遍历每个样本，计算 MIoU 当前值，并累加到 MIoU_sum
        for i in range(pred_inst.shape[0]):
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())

            # 计算当前样本的 MIoU
            MIoU_current = 100.0 * \
                len(pred_inst_set.intersection(target_inst_set)) / \
                len(pred_inst_set.union(target_inst_set))

            # 累加 MIoU 当前值
            MIoU_sum += MIoU_current

            # 清空集合以供下一个样本使用
            pred_inst_set.clear()
            target_inst_set.clear()

        # 计算 MIoU2 的平均值
        MIoU2 = MIoU_sum / batch_size

        # 返回计算结果
        return res[0], trajectory_success_rate, MIoU1, MIoU2,correct_all


def test_inference(val_loader, model, args):
    model.eval()
    acc_top1 = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU1_meter = AverageMeter()
    MIoU2_meter = AverageMeter()

    # A0_acc = AverageMeter()
    # AT_acc = AverageMeter()
    CorrectAll = []
    for i in range(args.horizon):
        CorrectAll.append(AverageMeter())
    for i_batch, sample_batch in enumerate(val_loader):
        # compute output
        global_img_tensors = sample_batch[0].cuda().contiguous()
        # print('global_img_tensors')
        # print(global_img_tensors)
        video_label = sample_batch[1].cuda()
        # print('video_label')
        # print(video_label)

        batch_size_current, T = video_label.size()  # batch size(256), horizon steps

        # print('sample_batch[2]')
        # print(sample_batch[2])

        task_class = sample_batch[2].view(-1).cuda()
        # print('task_class')
        # print(task_class)

        cond = {}

        with torch.no_grad():
            cond[0] = global_img_tensors[:, 0, :].float()
            # print('cond[0]')
            # print(cond[0])
            cond[T - 1] = global_img_tensors[:, -1, :].float()
            # print('cond[T - 1]')
            # print(cond[T - 1])

            task_onehot = torch.zeros((task_class.size(0), args.class_dim))
            # [bs*T, ac_dim]
            ind = torch.arange(0, len(task_class))
            # print('ind')
            # print(ind)
            task_onehot[ind, task_class] = 1.

            task_onehot = task_onehot.cuda()
            # print('task_onehot')
            # print(task_onehot.size())  # torch.Size([256, 18])
            # print(task_onehot)
            temp = task_onehot.unsqueeze(1)
            task_class_ = temp.repeat(1, T, 1)  # [bs, T, args.class_dim]

            cond['task'] = task_class_
            # print('task_class_')
            # print(task_class_.size()) # torch.Size([256, 3, 18])
            # print(task_class_)

            # flatten the video labels
            video_label_reshaped = video_label.view(-1)
            
            # if args.if_jump == 1:
            output = model(cond, if_jump=True)
            # else:
            #     output = model(cond, if_jump=False)
            # print('output')
            # print(output.size())  # torch.Size([256, 3, 1659])
            # print(output)

            # .contiguous()用于确保Tensor的数据在内存中是连续存储的
            actions_pred = output.contiguous()
            actions_pred = actions_pred[:, :,
                                        args.class_dim:args.class_dim + args.action_dim].contiguous()
            # print('actions_pred')
            # print(actions_pred.size())  # torch.Size([256, 3, 105])
            # print(actions_pred)
            actions_pred = actions_pred.view(-1, args.action_dim)

            # print('actions_pred')
            # print(actions_pred.size())  # torch.Size([768, 105])
            # print(actions_pred)
            # print()
            # print('video_label_reshaped')
            # print(video_label_reshaped.size())  # torch.Size([768])
            # print(video_label_reshaped)

            acc1, trajectory_success_rate, MIoU1, MIoU2, correct_all = \
                accuracy2(actions_pred.cpu(), video_label_reshaped.cpu(),
                          topk=(1,), max_traj_len=args.horizon)

        acc_top1.update(acc1.item(), batch_size_current)
        trajectory_success_rate_meter.update(
            trajectory_success_rate.item(), batch_size_current)
        MIoU1_meter.update(MIoU1, batch_size_current)
        MIoU2_meter.update(MIoU2, batch_size_current)
        
        for i in range(len(correct_all)):
            CorrectAll[i].update(correct_all[i], batch_size_current)
            
        for i in range(len(CorrectAll)):
            CorrectAll[i] = torch.tensor(CorrectAll[i].avg)

    return torch.tensor(acc_top1.avg), \
        torch.tensor(trajectory_success_rate_meter.avg), \
        torch.tensor(MIoU1_meter.avg), torch.tensor(MIoU2_meter.avg), CorrectAll


def reduce_tensor(tensor):
    if dist.is_initialized():
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    else:
        return tensor


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    env_dict = get_environment_shape(args.dataset, args.horizon,args.base_model)
    args.action_dim = env_dict['action_dim']
    args.observation_dim = env_dict['observation_dim']
    args.class_dim = env_dict['class_dim']
    args.root = env_dict['root']
    args.json_path_train = env_dict['json_path_train']
    args.json_path_val = env_dict['json_path_val']
    args.json_path_val2 = env_dict['json_path_val2']
    args.n_diffusion_steps = env_dict['n_diffusion_steps']
    args.n_train_steps = env_dict['n_train_steps']
    # epoch_env = env_dict['epochs']
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
    # print('ngpus_per_node:', ngpus_per_node)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # print('gpuid:', args.gpu)

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

    # Test data loading code
    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset)
        test_sampler.shuffle = False
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )

    # create model
    if args.base_model == 'base':
        temporal_model = temporal.TemporalUnet(args,
                                               dim=256,
                                               dim_mults=(1, 2, 4), )
    elif args.base_model == 'predictor':
        temporal_model = temporalPredictor.TemporalUnet(args,
                                                        dim=256,
                                                        dim_mults=(1, 2, 4), )

    diffusion_model = diffusion.GaussianDiffusion(
        args, temporal_model)

    model = utils.Trainer(args, diffusion_model, None)

    # if args.pretrain_cnn_path:
    #     net_data = torch.load(args.pretrain_cnn_path)
    #     model.model.load_state_dict(net_data)
    #     model.ema_model.load_state_dict(net_data)
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

    if args.resume:
        checkpoint_path = args.ckpt_path
        if checkpoint_path:
            print("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(
                checkpoint_path, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"], strict=True)
            model.ema_model.load_state_dict(checkpoint["ema"], strict=True)
            model.step = checkpoint["step"]
        else:
            assert 0

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    time_start = time.time()
    acc_top1_reduced_sum = []
    trajectory_success_rate_meter_reduced_sum = []
    MIoU1_meter_reduced_sum = []
    MIoU2_meter_reduced_sum = []
    # acc_a0_reduced_sum = []
    # acc_aT_reduced_sum = []
    test_times = 1

    for epoch in range(0, test_times):
        tmp = args.seed
        random.seed(tmp)
        np.random.seed(tmp)
        torch.manual_seed(tmp)
        torch.cuda.manual_seed_all(tmp)

        acc_top1, trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, correct_all = test_inference(
            test_loader, model.ema_model, args)

        acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
        trajectory_success_rate_meter_reduced = reduce_tensor(
            trajectory_success_rate_meter.cuda()).item()
        MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
        MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
        # for i in range(len(correct_all)):
        #     correct_all[i] = reduce_tensor(correct_all[i].cuda()).item()
        # acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
        # acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

        acc_top1_reduced_sum.append(acc_top1_reduced)
        trajectory_success_rate_meter_reduced_sum.append(
            trajectory_success_rate_meter_reduced)
        MIoU1_meter_reduced_sum.append(MIoU1_meter_reduced)
        MIoU2_meter_reduced_sum.append(MIoU2_meter_reduced)
        CorrectAll = []
        for i in range(len(correct_all)):
            CorrectAll.append(reduce_tensor(correct_all[i].cuda()).item())
        # acc_a0_reduced_sum.append(acc_a0_reduced)
        # acc_aT_reduced_sum.append(acc_aT_reduced)

    if args.rank == 0:
        time_end = time.time()
        print('time: ', time_end - time_start)
        print('-----------------Mean&Var-----------------------')
        print('Val/EpochAcc@1', sum(acc_top1_reduced_sum) /
              test_times, np.var(acc_top1_reduced_sum))
        print('Val/Traj_Success_Rate', sum(trajectory_success_rate_meter_reduced_sum) /
              test_times, np.var(trajectory_success_rate_meter_reduced_sum))
        print('Val/MIoU1', sum(MIoU1_meter_reduced_sum) /
              test_times, np.var(MIoU1_meter_reduced_sum))
        print('Val/MIoU2', sum(MIoU2_meter_reduced_sum) /
              test_times, np.var(MIoU2_meter_reduced_sum))
        # print('Val/acc_a0', sum(acc_a0_reduced_sum) /
        #       test_times, np.var(acc_a0_reduced_sum))
        # print('Val/acc_aT', sum(acc_aT_reduced_sum) /
        #       test_times, np.var(acc_aT_reduced_sum))
        for i in range(len(CorrectAll)):
            print(f'Val/CorrectAll_{i}', sum(CorrectAll[i]) /
              test_times, np.var(CorrectAll[i]))


if __name__ == "__main__":
    main()
