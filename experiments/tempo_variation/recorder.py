"""Owned update packets: optimizer state before/after, gradients and draw RNG."""
import numpy as np
import torch

from .data import require, sha


def snapshot(path, model, optimizer, rng):
    # This is a generated research artifact, not a loadable public model format.
    with path.open('xb') as stream:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []), stream)
    return sha(path)


def gradient_vector(model):
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), 'invalid gradient')
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu().numpy().copy()


def gradients(path, before, after):
    require(before.shape == after.shape and np.isfinite(before).all() and np.isfinite(after).all(), 'invalid recorded gradients')
    with path.open('xb') as stream:
        np.savez(stream, unclipped=before, clipped=after)
    return sha(path)


def adamw_reference(before, after, gradient):
    """Independent NumPy float64 formula; never invokes an optimizer."""
    groups = before['optimizer']['param_groups']
    require(len(groups) == 1 and groups == after['optimizer']['param_groups'], 'optimizer configuration changed')
    group = groups[0]
    require(not group['amsgrad'] and not group['maximize'], 'unsupported optimizer variant')
    names = list(before['model'])
    require(names == list(after['model']) and len(names) == len(group['params']), 'parameter geometry changed')
    beta1, beta2 = group['betas']
    cursor, maximum = 0, 0.
    for name, index in zip(names, group['params']):
        w = before['model'][name].numpy().astype(np.float64)
        size = w.size
        g = np.asarray(gradient[cursor:cursor+size], np.float64).reshape(w.shape)
        cursor += size
        old = before['optimizer']['state'].get(index)
        new = after['optimizer']['state'][index]
        step = int(old['step'])+1 if old is not None else 1
        require(int(new['step']) == step, 'optimizer step count differs')
        m = beta1*old['exp_avg'].numpy().astype(np.float64)+(1-beta1)*g if old is not None else (1-beta1)*g
        v = beta2*old['exp_avg_sq'].numpy().astype(np.float64)+(1-beta2)*g*g if old is not None else (1-beta2)*g*g
        expected = w*(1-group['lr']*group['weight_decay'])-group['lr']*(m/(1-beta1**step))/(np.sqrt(v/(1-beta2**step))+group['eps'])
        actual = after['model'][name].numpy().astype(np.float64)
        require(np.isfinite(expected).all() and np.isfinite(actual).all(), 'nonfinite update')
        error = float(np.max(np.abs(actual-expected)))
        require(error <= 2e-7, 'saved AdamW displacement differs from reference')
        for observed, calculated in ((new['exp_avg'], m), (new['exp_avg_sq'], v)):
            require(np.allclose(observed.numpy(), calculated, rtol=2e-5, atol=2e-8), 'saved optimizer moment differs')
        maximum = max(maximum, error)
    require(cursor == len(gradient), 'gradient geometry differs')
    return maximum
