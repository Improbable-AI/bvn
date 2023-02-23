import torch
import torch.backends.cudnn as cudnn

# delete all these
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    cudnn.benchmark = True
    print('Using CUDA ..')


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_from_numpy(np_dict):
    return {
        k: from_numpy(v) for k, v in np_dict.items()
    }


def zeros(sizes, **kwargs):
    return torch.zeros(sizes, **kwargs).float().to(device)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).float().to(device)


def ones(sizes, **kwargs):
    return torch.ones(sizes, **kwargs).float().to(device)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs).float().to(device)


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(device)


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


def dict_detach_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }


def dict_to_tensor(np_dict):
    return {
        k: to_tensor(v) for k, v in np_dict.items()
    }


def to_tensor(*args, **kwargs):
    return torch.as_tensor(*args, **kwargs).float().to(device)


from functools import wraps
from contextlib import contextmanager, ExitStack
from torch import nn


@contextmanager
def Eval(*modules):
    """
    Context Manager for setting network to evaluation mode.

        Useful for ResNet (containing BatchNorm)

    :param modules: position arguments as network modules.
    :return:
    """
    train_modes = [m.training for m in modules]
    try:
        for m in modules:
            m.train(False)
        yield modules
    finally:
        for m, mode in zip(modules, train_modes):
            m.train(mode)


class RMSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Root Mean-square loss

        Creates a criterion that measures the root mean squared error (squared L2 norm) between each element in the input x and target y.

        The loss can be described as:

        ℓ(x, y) = L = {l1, …, lN}⊤,  ln = (xn − yn)2, 
        where N is the batch size. If reduce is True, then:
        ℓ(x, y) = ⎧⎨⎩\operatornamemean(L),   if size_average = True,         \operatornamesum(L),   if size_average = False. 

        The sum operation still operates over all the elements, and divides by n.
        The division by n can be avoided if one sets size_average to False.
        To get a batch of losses, a loss per batch element, set reduce to False. These losses are not averaged and are not affected by size_average.

        Shape:

        Input: (N, *) where * means, any number of additional dimensions
        Target: (N, *), same shape as the input
        eps: machine precision constant

        Examples:

        .. code::

            loss = nn.RMSLoss()
            input = torch.randn(3, 5, requires_grad=True)
            target = torch.randn(3, 5)
            output = loss(input, target)
            output.backward()
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(√MSELoss(ˆy, y) + eps)"


class View(nn.Module):
    def __init__(self, *size):
        """reshape input tensor

        :param size: reshapes size of tensor to [batch, *size]
        """
        super().__init__()
        self.size = size

    def forward(self, x):
        try:
            return x.view(-1, *self.size)
        except RuntimeError:
            print('Check if original input has its size changed.')

    def __repr__(self):
        return f"View(-1, {', '.join([str(n) for n in self.size])})"


class SmartView(nn.Module):
    _shape = None

    def __init__(self, last):
        """
        reshape input tensor

        :param range: int reshapes size of tensor to [batch, prod(x.shape[last:])]
        """
        super().__init__()
        self.last = last

    def forward(self, x):
        self._shape = x.shape[self.last:]
        return x.view(-1, np.prod(x.shape[self.last:]))

    def __repr__(self):
        if self._shape:
            return f"View(-1, {'* '.join([str(n) for n in self._shape])})"
        return f"View(-1, prod(x.shape[last:]))"


class Λ(nn.Module):
    def __init__(self, fn):
        """
        reshape nn module.

        :param size: reshapes size of tensor to [batch, *size]
        """
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        from textwrap import dedent
        from inspect import getsource
        return f"Λ({dedent(getsource(self.fn))})"

    def __getstate__(self):
        import dill
        return dill.dumps(self.fn)

    def __setstate__(self, state):
        import dill
        self.fn = dill.loads(state)


def module_device(module):
    if hasattr(module, 'parameters'):
        params = list(module.parameters())
        return params[0].device if params else None
    return None


def _torchify(fn, device=None, dtype=None, input_only=False, method=False, with_eval=False):
    """
    wraps function, turn inputs into torch tensors. Return values as numpy
    :param fn:
    :param device: Optional. Automatically assigns input tensor to the
                   device of the first parameter of the module.
    :param dtype: The reason why we tend to get type error is because default
                 numpy tensors are double, whereas torch defaults to single.
                 torch.tensor try to respect the number dtype.
    :param input_only: flag for not numpify the return value
    :param method: one of [bool, nn.Module] flag for application to hounded class methods.
        if a module is passed, eval with that module
    :return:
    """
    import torch
    import numpy as np

    device = device or module_device(fn)

    # need to add support for class methods.
    @wraps(fn)
    def wrapping(*args, **kwargs):
        _d = device() if callable(device) else device
        if method:
            cls_self, *args = args
            _cls = (cls_self,)
        else:
            _cls = tuple()

        if with_eval is False:
            module = None
        else:
            module = with_eval if isinstance(with_eval, torch.nn.Module) else fn

        with Eval(module) if module else ExitStack():
            r = fn(*_cls, *[torch.tensor(arg, device=_d, dtype=dtype)
                            if isinstance(arg, np.ndarray) or isinstance(arg, list) else arg
                            for arg in args],
                   **{k: torch.tensor(v, device=_d, dtype=dtype)
                   if isinstance(v, np.ndarray) or isinstance(v, list) else v
                      for k, v in kwargs.items()})

        # we return numpy arrays.
        return r if input_only else r.detach().cpu().numpy()

    # allow training as model
    for k in dir(fn):
        if k.startswith("_"):
            continue
        setattr(wrapping, k, getattr(fn, k))

    # pass the original in as _unwrap
    wrapping.module = fn.module if hasattr(fn, "module") else fn
    # superseded by __wrapped__ attribute in PY3.
    wrapping.unwrap = fn  # unwrap is not recursive.

    return wrapping


def torchify(fn=None, device=None, dtype=None, input_only=False, method=False, with_eval=None):
    """
    wraps function, turn inputs into torch tensors. Return values as numpy
    :param fn:
    :param device: Optional. Automatically assigns input tensor to the
                   device of the first parameter of the module.
    :param dtype: The reason why we tend to get type error is because default
                 numpy tensors are double, whereas torch defaults to single.
                 torch.tensor try to respect the number dtype.
    :param input_only: flag for not numpify the return value
    :param method: bool flag for application to hounded class methods.
    :param eval_mode: bool flag to use Eval context
    :return:
    """
    if callable(fn):
        return _torchify(fn, device=device, dtype=dtype, input_only=input_only, method=method, with_eval=with_eval)
    return lambda fn: _torchify(fn, device=device, dtype=dtype, input_only=input_only, method=method)


if __name__ == "__main__":
    import torch
    import numpy as np

    lam = lambda x: x

    _ = torchify(lam, dtype=torch.float32)
    print(_(np.zeros(10, dtype=np.uint8)).dtype)
