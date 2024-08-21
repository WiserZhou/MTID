import torch


def accuracy(output, target, topk=(1,), max_traj_len=0):
    # Ensure gradients are not calculated for this section
    with torch.no_grad():
        # Determine the maximum value of topk
        maxk = max(topk)
        # Get the batch size from the target tensor
        batch_size = target.size(0)

        # Get the top-k predictions for each output in the batch
        _, pred = output.topk(maxk, 1, True, True)
        # Transpose the predictions to match the expected shape
        pred = pred.t()  # [k, bs*T]
        # Check which predictions are correct by comparing with the target
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [k, bs*T]

        # Reshape to consider each trajectory separately
        correct_a = correct[:1].view(-1, max_traj_len)  # [bs, T]
        # Calculate the accuracy at the first time step
        correct_a0 = correct_a[:, 0].reshape(-1).float().mean().mul_(100.0)
        # Calculate the accuracy at the last time step
        correct_aT = correct_a[:, -1].reshape(-1).float().mean().mul_(100.0)

        # Initialize a list to store results for each k in topk
        res = []
        for k in topk:
            # Calculate the number of correct predictions in the top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # Convert to percentage and append to results
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]  # (1, bs*T)

        # Success Rate: Consider a trajectory successful if all its actions are correct
        trajectory_success = torch.all(correct_1.view(
            correct_1.shape[1] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / \
            trajectory_success.shape[0]

        # Mean Intersection over Union (MIoU)
        # Get the top-1 predictions for each output in the batch
        _, pred_token = output.topk(1, 1, True, True)  # [bs*T, 1]
        # Reshape predictions to match the batch size
        pred_inst = pred_token.view(correct_1.shape[1], -1)  # [bs*T, 1]
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[1], -1)  # [bs*T, 1]
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            # Collect sets of predicted and target actions
            pred_inst_set.add(tuple(pred_inst[i].tolist()))
            target_inst_set.add(tuple(target_inst[i].tolist()))
        # Calculate IoU for all actions
        MIoU1 = 100.0 * len(pred_inst_set.intersection(target_inst_set)
                            ) / len(pred_inst_set.union(target_inst_set))

        # Reshape to consider each trajectory separately
        batch_size = batch_size // max_traj_len
        pred_inst = pred_token.view(batch_size, -1)  # [bs, T]
        pred_inst_set = set()
        target_inst = target.view(batch_size, -1)  # [bs, T]
        target_inst_set = set()
        MIoU_sum = 0
        for i in range(pred_inst.shape[0]):
            # Update sets for each trajectory
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())
            # Calculate IoU for the current trajectory
            MIoU_current = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(
                pred_inst_set.union(target_inst_set))
            MIoU_sum += MIoU_current
            # Clear sets for the next iteration
            pred_inst_set.clear()
            target_inst_set.clear()

        # Average IoU over all trajectories
        MIoU2 = MIoU_sum / batch_size

        # Return results including accuracy at top-k, success rate, and MIoU
        return res, trajectory_success_rate, MIoU1, MIoU2, correct_a0, correct_aT
