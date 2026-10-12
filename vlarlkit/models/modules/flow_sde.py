import math

import torch


PPO_NOISE_LEVEL = 0.5


def leave_one_out_center(values):
    k = values.shape[-1]
    assert k > 1
    return (k / (k - 1)) * (values - values.mean(dim=-1, keepdim=True))


def stream_seed(seed, update_step, rank, world_size, stream):
    assert stream in ('node', 'noise') and 0 <= rank < world_size
    return 2 * (int(seed) + int(update_step) * world_size + rank) + (stream == 'noise')


def node_plan(seed, update_step, rank, world_size, local_batch, num_steps):
    key = stream_seed(seed, update_step, rank, world_size, 'node')
    generator = torch.Generator(device='cpu').manual_seed(key)
    return torch.randint(num_steps, (local_batch,), generator=generator), key


@torch.no_grad()
def noise_plan(seed, update_step, rank, world_size, local_batch, paths,
               horizon, action_dim, device):
    key = stream_seed(seed, update_step, rank, world_size, 'noise')
    generator = torch.Generator(device=device).manual_seed(key)
    shape = (local_batch, paths, horizon, action_dim)
    return (torch.randn(shape, generator=generator, device=device),
            torch.randn(shape, generator=generator, device=device), key)


def chain_coefficients(indices, num_steps=5, noise_level=PPO_NOISE_LEVEL):
    """Exact PPO float32 time grid and actor/reference Gaussian coefficients."""
    assert num_steps == 5 and noise_level == PPO_NOISE_LEVEL
    grid = torch.linspace(1, 1 / num_steps, num_steps, device=indices.device)
    grid = torch.cat([grid, grid.new_zeros(1)])
    native_times = grid[indices]
    steps = grid[indices] - grid[indices + 1]
    adjusted = torch.where(grid == 1, grid[1], grid)
    sigmas = (noise_level * torch.sqrt(grid / (1 - adjusted)))[:-1][indices]
    shrink = sigmas.square() / (2 * native_times)
    scale = 1 + shrink * (1 - native_times)
    # Preserve the QRFlow backend's existing velocity/time convention.
    times = indices.float() / num_steps
    return times, sigmas, scale, steps, shrink


def transition_mean(x, velocity, indices, num_steps=5, noise_level=PPO_NOISE_LEVEL):
    _, _, scale, steps, shrink = chain_coefficients(indices, num_steps, noise_level)
    return ((1 - steps * shrink)[..., None, None] * x
            + (steps * scale)[..., None, None] * velocity)


def gaussian_log_ratio(control, noise, h):
    h = torch.as_tensor(h, dtype=control.dtype, device=control.device)
    return .5 * h * control.square().sum((-2, -1)) + h.sqrt() * (control * noise).sum((-2, -1))


def selected_transition(x, fast, reference, innovation, indices, mask):
    _, sigma, scale, steps, _ = chain_coefficients(indices)
    reference_mean = transition_mean(x, reference, indices)
    control = scale[..., None, None] * (fast - reference) / sigma[..., None, None]
    gaussian_mean = reference_mean + (steps * sigma)[..., None, None] * control
    gaussian_next = gaussian_mean + (steps.sqrt() * sigma)[..., None, None] * innovation.float()
    return torch.where(mask[..., None, None], gaussian_next, x + .2 * fast), reference_mean, control


@torch.no_grad()
def sample_single_sde(velocity, initial, innovation, nodes):
    """The collection path uses the same selected transition as actor training."""
    assert initial.ndim == 4 and initial.shape == innovation.shape
    batch, count = initial.shape[:2]
    assert nodes.shape == (batch,) and bool(((nodes >= 0) & (nodes < 5)).all())
    value = initial.float()
    chosen = nodes.to(value.device)[:, None].expand(batch, count)
    for node in range(5):
        indices = torch.full((batch, count), node, device=value.device, dtype=torch.long)
        times = chain_coefficients(indices)[0]
        fast, reference = velocity(value, times, False).float(), velocity(value, times, True).float()
        value, _, _ = selected_transition(value, fast, reference, innovation, indices, chosen == node)
    return value


@torch.no_grad()
def prepare_actor_targets(velocity, utility, initial, innovation, nodes, beta,
                          noise_level=PPO_NOISE_LEVEL):
    assert initial.ndim == 4 and initial.shape == innovation.shape
    b, k = initial.shape[:2]
    assert k > 1 and nodes.shape == (b,)
    assert bool(((nodes >= 0) & (nodes < 5)).all())
    assert math.isfinite(beta) and beta > 0 and noise_level == PPO_NOISE_LEVEL
    h = .2
    x = initial.float()
    chosen = nodes.to(x.device)[:, None].expand(b, k)
    selected = None
    selected_ratio = x.new_zeros((b, k))
    selected_kl = x.new_zeros((b, k))
    for i in range(5):
        index = torch.full((b, k), i, device=x.device, dtype=torch.long)
        times, sigma, scale, steps, _ = chain_coefficients(index, 5, noise_level)
        fast = velocity(x, times, False).float()
        reference = velocity(x, times, True).float()
        next_x, reference_mean, control = selected_transition(x, fast, reference, innovation, index, chosen == i)
        mask = chosen == i
        ratio = gaussian_log_ratio(control, innovation, steps)
        selected_ratio = torch.where(mask, ratio, selected_ratio)
        selected_kl = torch.where(mask, .5 * steps * control.square().sum((-2, -1)), selected_kl)
        node_target = dict(positions=x, positions_next=next_x, behavior_velocities=fast,
                           references=reference,
                           reference_means=reference_mean, time_indices=index,
                           times=times, sigmas=sigma, step_sizes=steps)
        if selected is None:
            selected = {key: torch.zeros_like(value) for key, value in node_target.items()}
        for key, value in node_target.items():
            selected[key] = torch.where(mask.reshape(b, k, *([1] * (value.ndim - 2))),
                                        value, selected[key])
        x = next_x
    reward = utility(x).float()
    assert reward.shape == (b, k) and bool(torch.isfinite(reward).all())
    assert bool(((reward >= -10) & (reward <= 10)).all())
    raw_cost = leave_one_out_center(selected_ratio - beta * reward)
    selected['cost_to_go'] = raw_cost[None].detach()
    targets = {key: value[None].detach() if key != 'cost_to_go' else value
               for key, value in selected.items()}
    metrics = dict(q_margin=float(reward.mean()), control_cost=float(selected_kl.mean()))
    return targets, metrics


def qrflow_loss(fast_velocity, targets, num_steps, noise_level=PPO_NOISE_LEVEL):
    assert num_steps == 5 and noise_level == PPO_NOISE_LEVEL
    assert targets['positions'].shape[0] == 1
    h = .2
    targets = {key: value.detach() for key, value in targets.items()}
    _, sigma, scale, steps, _ = chain_coefficients(targets['time_indices'], 5, noise_level)
    control = scale[..., None, None] * (fast_velocity.float() - targets['references']) / sigma[..., None, None]
    mean = targets['reference_means'] + (steps * sigma)[..., None, None] * control
    noise = (targets['positions_next'] - mean) / (steps.sqrt() * sigma)[..., None, None]
    log_probability = -.5 * noise.square().sum((-2, -1))
    # Uniform j defines an average over five conditional policies: no N/M factor.
    return (2 / h) * (targets['cost_to_go'] * log_probability).sum(0).mean()
