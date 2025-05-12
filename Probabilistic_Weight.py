import torch

def random_sampling(k,num_samples):
    indices = torch.randperm(k.size(2))[:num_samples]

    x_sampled = torch.zeros_like(k)

    x_sampled[:, :, indices, :] = k[:, :, indices, :]

    return x_sampled

def S_weight(tensor1):
    max_values, _ = torch.max(tensor1, dim=3, keepdims=True)
    mean_values = torch.mean(tensor1, dim=3, keepdims=True)

    new_values = max_values - mean_values
    tensor = torch.zeros_like(tensor1)
    tensor[:, :, :, :] = new_values

    return tensor

def Top_u(q,m,num_samples,W):
    topk_values, topk_indices = torch.topk(m, num_samples, dim=-2, largest=True)
    topk_indices1 = topk_indices[:, :, :, 1]
    output_tensor = torch.zeros_like(q)

    expanded_indices = topk_indices1.unsqueeze(-1).expand(-1, -1, -1, W)
    result = torch.gather(q, 2, expanded_indices)
    output_tensor.scatter_(2, expanded_indices, result)

    return output_tensor, result, expanded_indices
