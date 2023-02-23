def proc_id():
    try:
        print('proc_id')
        from mpi4py import MPI
    except ImportError:
        return 0
    return MPI.COMM_WORLD.Get_rank()
