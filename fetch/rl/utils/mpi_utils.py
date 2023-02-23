import numpy as np
import torch


def use_mpi():
    print('use_mpi')
    return get_size() > 1


def is_root():
    return True
    # print("is_root")
    # return get_rank() == 0


def global_mean(x):
    print("mpi_average")
    from mpi4py import MPI

    global_x = np.zeros_like(x)
    comm = MPI.COMM_WORLD
    comm.Allreduce(x, global_x, op=MPI.SUM)
    global_x /= comm.Get_size()
    return global_x


def global_sum(x):
    print('global_sum')
    from mpi4py import MPI

    global_x = np.zeros_like(x)
    comm = MPI.COMM_WORLD
    comm.Allreduce(x, global_x, op=MPI.SUM)
    return global_x


def bcast(x):
    print('bcast')
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm.Bcast(x, root=0)
    return x


def get_rank():
    print('get_rank')
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_rank()


def get_size():
    print('get_size')
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_size()


def sync_networks(network):
    print('sync_networks')
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params')
    comm.Bcast(flat_params, root=0)
    _set_flat_params_or_grads(network, flat_params, mode='params')


def sync_grads(network, scale_grad_by_procs=True):
    print('sync_grads')
    from mpi4py import MPI
    flat_grads = _get_flat_params_or_grads(network, mode='grads')
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    if scale_grad_by_procs:
        global_grads /= comm.Get_size()
    _set_flat_params_or_grads(network, global_grads, mode='grads')


def _get_flat_params_or_grads(network, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    attr = 'data' if mode == 'params' else 'grad'
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
