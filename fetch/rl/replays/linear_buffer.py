"""
The goal of this buffer implementation is to build a generalizable
linearized buffer that can adapt to a wide variety of usages patterns
using query strings and some form of DSL.

Example usage include:

buffer.extend(traj, "some_key", {"next@img": lambda img: img[1:]}).
"""
from typing import Sequence, NamedTuple

from collections import defaultdict, deque
import numpy as np


class ArrayDict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        kwargs.update(d or {})
        for k, v in kwargs.items():
            self[k] = np.array(v)

    def __getitem__(self, item):
        if type(item) is str:
            return dict.__getitem__(self, item)

        _ = {}
        for k, v in self.items():
            v_ = np.array(v)
            if isinstance(item, int):
                _ind = item
            elif isinstance(item, float):
                # todo: add float support to slice.
                # todo: need to check float in [0, 1).
                from math import floor
                _ind = floor(item * len(self))
            elif isinstance(item, slice):
                _ind = item
            elif isinstance(item, tuple):
                _ind = item
            else:
                _ind = np.broadcast_to(item, v_.shape)
            _[k] = v_[_ind]
        return ArrayDict(_)


class BaseBuffer:
    """The key idea of this buffer implementation is to keep all experience in a linear array.

    We do sampling and processing off this linear array structure."""

    def __init__(self, maxlen):
        self.__len = 0
        self.maxlen = maxlen
        self.buffer = defaultdict(lambda: deque(maxlen=maxlen))

    def state_dict(self):
        return dict(buffer=self.buffer.copy())

    def load_state_dict(self, d):
        for k, v in d.items():  # changes the reference
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)}, maxlen={self.maxlen}, {list(self.buffer.keys())})"

    def add(self, **kwargs):
        self.extend(**{k: [v] for k, v in kwargs.items()})

    def extend(self, **kwargs):
        """Saves a transition."""
        # TODO: add assert to check lens match
        for k, v in kwargs.items():
            self.buffer[k].extend(v)
        self.__len = len(self.buffer[k])

    def sample(self, batch_size: int, **__):
        """
        if batch_size larger than buffer length, returns buffer length

        :param batch_size: int, size for the sample batch.
        :return: dict
        """
        # need to change this to a generator. sample without replacement.
        batch = {}
        inds = np.random.rand(len(self)).argsort()[:batch_size]
        for i, (k, v) in enumerate(self.buffer.items()):
            _ = np.take(v, inds, axis=0)  # have to specify axis, otherwise flattens array.
            try:
                batch[k] = np.stack(_, axis=0)
            except ValueError:  # usually happens when the values are different shape.
                batch[k] = _
        return batch

    def sample_all(self, batch_size, *keys, proc=None, **proc_opts):
        """Samples the entire buffer without replacement.

        :param batch_size:
        :param keys: "perm@img" gives the permutation
        :param proc:
        :param proc_opts: key-value passed into the collate function
        :return:
        """
        if not len(self):
            return

        rand_ind = np.argsort(np.random.rand(self.__len))

        buffer = {k: np.array(v) for k, v in self.buffer.items()}

        keys = keys or list(self.buffer.keys())
        for ind_batch in np.array_split(rand_ind, np.ceil(len(self) / batch_size)):
            d = {k: buffer[k][ind_batch] for k in keys}
            if callable(proc):
                d = proc(**d, **proc_opts)
            if len(keys) == 1:
                yield d[keys[0]]
            else:
                yield [d[k] for k in keys]

    def __len__(self):
        return self.__len

    def __getitem__(self, inds):
        return ArrayDict(self.buffer)[inds]

    def clear(self):
        self.__len = 0
        self.buffer.clear()


class InfinityArray:
    """
    An infinite linear array that uses rolling `start` and `end`
    to indicate the relative position. This will overflow when
    python integer overflows, so there is not really a risk.

    Need to watch out for numpy int32 and long overflow at 2*64 - 1
    when np.int32 or np.long are used as indices besides this
    array.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.start = 0
        self.end = 0

    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)}, maxlen={self.maxlen}, {self.start}, {self.end})"

    def __getitem__(self, item):
        # todo: add advance indexing support
        if isinstance(item, int) or np.isscalar(item) and np.issubdtype(item.dtype, np.integer):
            if item >= self.end or item < self.start:
                raise IndexError(f"index {item} falls outside of the "
                                 f"range ({self.start}, {self.end})")
            ind = item + (- self.start if item >= 0 else self.end)
            return self.data[ind]
        elif isinstance(item, Sequence) or np.issubdtype(item.dtype, np.integer):
            return np.array([
                self[i] for i in item
            ])
        elif isinstance(item, slice):
            return self[list(range(self.start, self.end))]

    def __setitem__(self, key, value):
        raise NotImplemented("__setitem__ is not yet supported.")

    def extend(self, images):
        self.data.extend(images)
        self.end += len(images)
        self.start = self.end - len(self.data)


def to_float(device='cpu', **kwargs):
    """Use with functools.partial to pass the device parameter"""
    import torch
    return {k: torch.tensor(v, dtype=torch.float).to(device) for k, v in kwargs.items()}


class ImageBuffer(BaseBuffer):
    """
    Buffer that avoids duplicating the observations.

    All adding need to be batched.
    This is because the transitions are off by 1.
    """

    def __init__(self, maxlen):
        super().__init__(maxlen)
        self.images = InfinityArray(maxlen=maxlen * 2)

    @property
    def end(self):
        return self.images.end

    def add(self, image, **kwargs):
        self.extend([image], **{k: [v] for k, v in kwargs.items()})

    def extend(self, image, **kw_indices):
        """save the indices in the buffer."""
        self.images.extend(image)
        super().extend(**kw_indices)

    def sample_all(self, batch_size, *keys, proc=None, **proc_opts):
        stripped = tuple(k[1:] if k.startswith("@") else k for k in keys)
        for values in super().sample_all(batch_size, *stripped):
            values = [values] if len(keys) == 1 else values
            d = {k: self.images[values[i]] if k.startswith("@") else values[i] for i, k in enumerate(keys)}
            if callable(proc):
                d = proc(**d, **proc_opts)
            if len(keys) == 1:
                yield d[keys[0]]
            else:
                yield [d[k] for k in keys]


class IndexBuffer(BaseBuffer):
    def __init__(self, maxlen, *keys):
        BaseBuffer.__init__(self, maxlen)
        self.keys = keys
        for k in keys:
            setattr(self, k, InfinityArray(maxlen=maxlen * 4))

    @property
    def end(self):
        return getattr(self, self.keys[0]).end

    def add(self, *tensors, **kwargs):
        self.extend(*[[t] for t in tensors], **{k: [v] for k, v in kwargs.items()})

    def extend(self, *tensors, **kw_indices):
        """save the indices in the buffer."""
        for k, t in zip(self.keys, tensors):
            getattr(self, k).extend(t)
        super().extend(**kw_indices)

    def sample_all(self, batch_size, *keys, proc=None, **proc_opts):
        stripped = tuple(k.split('@')[-1] for k in keys)
        for values in super().sample_all(batch_size, *stripped):
            d = {}
            for k, v in zip(keys, [values] if len(keys) == 1 else values):
                if '@' in k:
                    buffer_key, index_key = k.split('@')
                    d[k] = getattr(self, buffer_key)[v]
                else:
                    d[k] = v
            if callable(proc):
                d = proc(**d, **proc_opts)
            if len(keys) == 1:
                yield d[keys[0]]
            else:
                yield [d[k] for k in keys]
