import torch
from common.utils import plot_batch_slices

def sample_next_slices(noised_target_slices, condition_slices, model, noise_scheduler, batch_size, device, encoder_hidden_states=None, verbose=False):

    target_slices = noised_target_slices

    # for t in noise_scheduler.timesteps.flip(0):
    for t in noise_scheduler.timesteps:
        if verbose:
            print(f't is {t}')
        timesteps = torch.tensor([t], device=device, dtype=torch.float32).repeat(batch_size)

        input_tensor = torch.concat((target_slices, condition_slices), dim=1)

        with torch.no_grad():
            predicted_noise = model(input_tensor, timesteps, encoder_hidden_states).sample

        target_slices = noise_scheduler.step(predicted_noise, t, target_slices).prev_sample

    return target_slices

def get_index_encoding(index_encoding_type, target_start_index, slices_at_once, embedder):

    if index_encoding_type is None:
        return None
    elif index_encoding_type == 'index':
        encoding = target_start_index
        return encoding.unsqueeze(1).unsqueeze(1).float()
    elif index_encoding_type == 'embedded':
        encoding = embedder(target_start_index//slices_at_once)
        return encoding.unsqueeze(1)


def sample_iteratively(condition_start_index, slices_at_once, condition_batch, model, noise_scheduler, embedder, batch_size, device, symmetric_normalisation, index_encoding_type=None, verbose=False):

    (min_val, max_val) = (-1, 1) if symmetric_normalisation else (0, 1)

    output_batch = torch.zeros(batch_size, 128, 128, 128)

    # no conditioning
    if condition_start_index is None:
        target_start_index = 0
        condition_slices = torch.zeros(batch_size, slices_at_once, 128, 128, device=device)
    else:
        target_start_index = condition_start_index + slices_at_once
        condition_slices = condition_batch[:, condition_start_index:target_start_index].float().to(device)
        output_batch[:, condition_start_index:target_start_index] = condition_slices
        print(f'Condition slices populated from slice {condition_start_index} to {target_start_index}.')
        plot_batch_slices(condition_slices, batch_size, slices=slices_at_once, vmin=min_val)

    for i in range(target_start_index, 128, slices_at_once):
        print(f'Sampling slices {i} to {i+slices_at_once}...')
        noised_target_slices = torch.randn(batch_size, slices_at_once, 128, 128, device=device)
        
        encoding = get_index_encoding(index_encoding_type, target_start_index=torch.full((batch_size,), i, dtype=torch.int, device=device), slices_at_once=slices_at_once, embedder=embedder)
        target_slices = sample_next_slices(noised_target_slices, condition_slices, model, noise_scheduler, batch_size, device, encoder_hidden_states=encoding)
        if verbose:
            print(f'Generated slices in range [{target_slices.min(), target_slices.max()}]')
        
        target_slices = torch.clip(target_slices, min=min_val, max=max_val)

        plot_batch_slices(target_slices, batch_size, slices=slices_at_once, vmin=min_val)

        output_batch[:, i:i+slices_at_once] = target_slices

        print(f'Complete.')
        condition_slices = target_slices.float()

    return output_batch