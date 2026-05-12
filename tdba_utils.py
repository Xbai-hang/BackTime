import numpy as np
import torch


def clamp_max_delay(pred_len, pattern_len, max_delay):
    return max(0, min(int(max_delay), int(pred_len) - int(pattern_len)))


def sample_tdba_delays(mode, fixed_delay, max_delay, var_num, device, variable_delays=None):
    max_delay = int(max_delay)
    if mode == 'random':
        return torch.randint(0, max_delay + 1, (var_num,), device=device)
    if mode == 'variable' and variable_delays:
        values = torch.tensor(variable_delays, dtype=torch.long, device=device)
        if values.numel() < var_num:
            values = values.repeat(int(np.ceil(var_num / values.numel())))
        return torch.clamp(values[:var_num], 0, max_delay)
    delay = max(0, min(int(fixed_delay), max_delay))
    return torch.full((var_num,), delay, dtype=torch.long, device=device)


def gaussian_position_guidance(delays, pred_len, sigma, device):
    d = torch.arange(pred_len, dtype=torch.float32, device=device).unsqueeze(1)
    centers = delays.float().unsqueeze(0)
    sigma = max(float(sigma), 1e-6)
    guidance = torch.exp(-((d - centers) ** 2) / (2 * sigma ** 2))
    return guidance / (guidance.sum(dim=0, keepdim=True) + 1e-8)


def delayed_region_loss(outputs, labels, atk_vars, delays, pattern_len, pred_len, offset_lambda=0.0):
    total = torch.zeros((), device=outputs.device)
    weight_sum = 0.0
    for s_id, var in enumerate(atk_vars):
        delay = int(delays[s_id].item())
        start = max(delay, 0)
        end = min(delay + pattern_len, pred_len)
        if end <= start:
            continue
        weight = float(np.exp(-float(offset_lambda) * delay))
        total = total + weight * torch.mean((outputs[:, start:end, var] - labels[:, start:end, var]) ** 2)
        weight_sum += weight
    if weight_sum == 0:
        return torch.zeros((), device=outputs.device)
    return total / weight_sum
