import copy
from model.helpers import AverageMeter, Logger
from .accuracy import *
import torch.distributed as dist


def cycle(dl):
    """
    Creates an infinite generator for iterating over a DataLoader.

    Args:
        dl (DataLoader): The dataloader to iterate over.

    Yields:
        Data batches from the dataloader.
    """
    while True:
        for data in dl:
            yield data


class EMA():
    """
    Implements the Exponential Moving Average (EMA) for smoothing model parameters.
    """

    def __init__(self, beta):
        """
        Initializes the EMA with a decay factor.

        Args:
            beta (float): The decay rate for the EMA.
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """
        Updates the moving average model's parameters with the current model's parameters.

        Args:
            ma_model (nn.Module): The EMA model to be updated.
            current_model (nn.Module): The current model providing new parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Computes the new average based on the old average and the new value.

        Args:
            old (Tensor): The previous average value.
            new (Tensor): The new value to update the average.

        Returns:
            Tensor: The updated average value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    """
    Manages the training process for a diffusion model including EMA, gradient accumulation, and scheduling.
    """
#  args.ema_decay, args.lr, args.gradient_accumulate_every,
#                           args.step_start_ema, args.update_ema_every, args.log_freq

    def __init__(
            self,
            args,
            diffusion_model,
            datasetloader,
    ):
        """
        Initializes the Trainer with the model, dataloader, and training parameters.

        Args:
            diffusion_model (nn.Module): The model to be trained.
            datasetloader (DataLoader): The dataloader providing training data.
            ema_decay (float): The decay rate for EMA.
            train_lr (float): The learning rate for the optimizer.
            gradient_accumulate_every (int): Steps to accumulate gradients before updating.
            step_start_ema (int): The step count to start EMA updates.
            update_ema_every (int): The frequency of EMA updates.
            log_freq (int): Frequency of logging during training.
        """
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(args.ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = args.update_ema_every

        self.step_start_ema = args.step_start_ema
        self.log_freq = args.log_freq
        self.gradient_accumulate_every = args.gradient_accumulate_every

        self.dataloader = cycle(datasetloader)
        self.optimizer = torch.optim.AdamW(
            diffusion_model.parameters(), lr=args.lr, weight_decay=0.0)

        self.reset_parameters()
        self.step = 0
        self.action_dim = args.action_dim

    def reset_parameters(self):
        """
        Resets the EMA model parameters to match the current model's parameters.
        """
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """
        Updates the EMA model parameters if the current step is greater than the start EMA step.
        """
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------- API Methods --------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps, if_calculate_acc, args, scheduler):
        """
        Trains the model for a specified number of steps.

        Args:
            n_train_steps (int): Number of training steps to perform.
            if_calculate_acc (bool): Flag indicating whether to compute accuracy metrics.
            args (Namespace): Additional arguments (dimensions, etc.) required for training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler for updating learning rate.

        Returns:
            Tuple: Contains loss and accuracy metrics if if_calculate_acc is True, otherwise only the loss.
        """
        self.model.train()
        self.ema_model.train()
        losses = AverageMeter()
        self.optimizer.zero_grad()

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                # Get the next batch of data
                # Batch(frames, labels, task)
                # Shape: [batch_size, (T+1), ob_dim]
                batch = next(self.dataloader)

                bs, T = batch[1].shape

                # contiguous():
                # Description: Ensures that the tensor's data is stored in a contiguous chunk of memory.
                # This is often necessary because some operations in PyTorch require that tensors be contiguous,
                # meaning their memory is laid out in a single, unbroken block.
                global_img_tensors = batch[0].cuda().contiguous().float()

                # set the start and end observation feature
                img_tensors = torch.zeros(
                    (bs, T, args.class_dim + args.action_dim + args.observation_dim))

                img_tensors[:, 0, args.class_dim +
                            args.action_dim:] = global_img_tensors[:, 0, :]  # Os
                img_tensors[:, -1, args.class_dim +
                            args.action_dim:] = global_img_tensors[:, -1, :]  # Og

                img_tensors = img_tensors.cuda()

                # Prepare labels and one-hot task encodings

                # Flatten video labels to a 1D tensor and move it to GPU. Shape: [batch_size * T]
                video_label = batch[1].view(-1).cuda()  # action label

                # Flatten task class labels to a 1D tensor and move it to GPU. Shape: [batch_size]
                task_class = batch[2].view(-1).cuda()

                # Initialize an empty tensor for one-hot encoded action labels.
                # If distributed training is initialized, use model.module.action_dim, else use model.action_dim.
                action_label_onehot = torch.zeros((video_label.size(0), self.action_dim))

                # Create an index tensor with values ranging from 0 to the length of video_label.
                ind = torch.arange(0, len(video_label))
                # Set the appropriate positions in the one-hot tensor to 1.
                # This creates one-hot encoded vectors for the action labels.
                # Fancy Indexing
                action_label_onehot[ind, video_label] = 1.

                # Reshape the one-hot encoded action labels to shape: [batch_size, T, action_dim] and move to GPU.
                action_label_onehot = action_label_onehot.reshape(
                    bs, T, -1).cuda()

                # Insert the action one-hot encodings into the corresponding section of img_tensors.
                # This integrates the action labels into the img_tensors.
                img_tensors[:, :, args.class_dim:args.class_dim +
                            args.action_dim] = action_label_onehot

                # Initialize an empty tensor for one-hot encoded task labels. Shape: [batch_size, class_dim]
                task_onehot = torch.zeros((task_class.size(0), args.class_dim))

                # Create an index tensor with values ranging from 0 to the length of task_class.
                ind = torch.arange(0, len(task_class))

                # Set the appropriate positions in the one-hot tensor to 1.
                # This creates one-hot encoded vectors for the task labels.
                task_onehot[ind, task_class] = 1.

                # Move the one-hot encoded task labels to GPU.
                task_onehot = task_onehot.cuda()

                # Add an extra dimension to the one-hot task tensor to match the sequence length.
                # Shape after unsqueeze: [batch_size, 1, class_dim]
                # unsqueeze(index):add a new dimension at index position
                temp = task_onehot.unsqueeze(1)

                # Repeat the one-hot task labels along the sequence length (T) dimension.
                # Shape after repeat: [batch_size, T, class_dim]
                task_class_ = temp.repeat(1, T, 1)

                # Insert the task one-hot encodings into the corresponding section of img_tensors.
                # This integrates the task labels into the img_tensors.
                img_tensors[:, :, :args.class_dim] = task_class_

                # Prepare the condition inputs
                # 0: start image,T-1:end image,task:task_class
                cond = {0: global_img_tensors[:, 0, :].float(), T - 1: global_img_tensors[:, -1, :].float(),
                        'task': task_class_}

                x = img_tensors.float()
                # print('start loss')

                # Compute the loss
                if dist.is_initialized():

                    loss = self.model.module.loss(x, cond)
                else:
                    loss = self.model.loss(x, cond)

                # print('loss complete')
                loss = loss / self.gradient_accumulate_every

                loss.backward()
                # print('backward')
                losses.update(loss.item(), bs)
                # print('update')
            
            if args.resume:
                # print('capturable')
                self.optimizer.param_groups[0]['capturable'] = True

            # Update model parameters and learning rate
            self.optimizer.step()

            self.optimizer.zero_grad()
            scheduler.step()

            # Update EMA model parameters
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.step += 1

        # Compute and return accuracy metrics if needed
        if if_calculate_acc:
            with torch.no_grad():
                output = self.ema_model(cond)

                if dist.is_initialized():
                    actions_pred = output[:, :, args.class_dim:args.class_dim+self.model.module.action_dim]\
                        .contiguous().view(-1, self.model.module.action_dim)  # Shape: [batch_size*T, action_dim]

                    (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                        accuracy(actions_pred.cpu(), video_label.cpu(), topk=(1, 5),
                                 max_traj_len=self.model.module.horizon)
                else:
                    actions_pred = output[:, :, args.class_dim:args.class_dim+self.model.action_dim]\
                        .contiguous().view(-1, self.model.action_dim)  # Shape: [batch_size*T, action_dim]

                    (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                        accuracy(actions_pred.cpu(), video_label.cpu(), topk=(1, 5),
                                 max_traj_len=self.model.horizon)

                return torch.tensor(losses.avg), acc1, acc5, torch.tensor(trajectory_success_rate), \
                    torch.tensor(MIoU1), torch.tensor(MIoU2), a0_acc, aT_acc

        else:
            return torch.tensor(losses.avg)
