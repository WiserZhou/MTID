import torch
import torch.nn.functional as F
import itertools

# Sample inputs

# Model outputs (logits) and targets (one-hot encoded)
# model_outputs = torch.randn(256, 3, 105)
# print(model_outputs)
# targets = F.one_hot(torch.randint(105, (256, 3)), num_classes=105).float()
# print(targets)

def compute_losses(model_outputs, targets, lambda_order=200.0, lambda_pos=0.01, lambda_perm=2.0):
    # Convert logits to probabilities
    probs = F.softmax(model_outputs, dim=-1)
    
    batch_size,horizon,action_dim = model_outputs.shape
    
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(model_outputs.view(-1, action_dim), targets.view(-1, action_dim).argmax(dim=-1))
    
    # Order Loss
    order_loss = 0
    for t in range(horizon - 1):
        a_t = targets[:, t].argmax(dim=-1)
        a_tp1 = targets[:, t + 1].argmax(dim=-1)
        
        prob_t_at = probs[torch.arange(batch_size), t, a_t]
        prob_tp1_atp1 = probs[torch.arange(batch_size), t + 1, a_tp1]
        
        order_loss += torch.max(torch.zeros_like(prob_t_at), prob_t_at - prob_tp1_atp1).mean()
    
    # Position Loss
    pos_loss = 0
    for t in range(horizon):
        predicted_label = probs[:, t].argmax(dim=-1)
        true_label = targets[:, t].argmax(dim=-1)
        pos_loss += torch.abs(predicted_label - true_label).float().mean()

    # Permutation Loss
    perm_loss = 0
    predicted_seq = probs.argmax(dim=-1)  # shape (batch_size, horizon)
    true_seq = targets.argmax(dim=-1)  # shape (batch_size, horizon)
    
    for i in range(batch_size):
        predicted_permutation = list(itertools.permutations(predicted_seq[i].tolist()))
        true_permutation = list(itertools.permutations(true_seq[i].tolist()))
        hamming_distances = [sum(p1 != p2 for p1, p2 in zip(predicted, true_seq[i].tolist())) 
                             for predicted in predicted_permutation]
        perm_loss += min(hamming_distances)
    perm_loss /= batch_size * horizon
    
    # Combine Losses
    total_loss = ce_loss + lambda_order * order_loss + lambda_pos * pos_loss + lambda_perm * perm_loss
    
    print(f"Total Loss: {total_loss.item()}")
    print(f"Cross Entropy Loss: {ce_loss.item()}")
    print(f"Order Loss: {lambda_order *order_loss.item()}")
    print(f"Position Loss: {lambda_pos * pos_loss.item()}")
    print(f"Permutation Loss: {lambda_perm * perm_loss}")  # Directly print perm_loss as it is a float
    
    return total_loss, ce_loss, order_loss, pos_loss, perm_loss

# Call the function
# total_loss, ce_loss, order_loss, pos_loss, perm_loss = compute_losses(model_outputs, targets, lambda_order=200, lambda_pos=0.01, lambda_perm=2)

# print(f"Total Loss: {total_loss.item()}")
# print(f"Cross Entropy Loss: {ce_loss.item()}")
# print(f"Order Loss: {order_loss.item()}")
# print(f"Position Loss: {pos_loss.item()}")
# print(f"Permutation Loss: {perm_loss}")  # Directly print perm_loss as it is a float
