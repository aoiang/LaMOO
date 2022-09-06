#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: To be removed once (if) https://github.com/pytorch/pytorch/pull/37385 lands

from __future__ import annotations

from collections import OrderedDict

import torch
from torch._six import container_abcs
from torch.nn import Module


class BufferDict(Module):
    r"""Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.BufferDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.BufferDict` (the argument to
      :meth:`~torch.nn.BufferDict.update`).

    Note that :meth:`~torch.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Args:
        buffers (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = nn.BufferDict({
                        'left': torch.randn(5, 10),
                        'right': torch.randn(5, 10)
                })

            def forward(self, x, choice):
                x = self.buffers[choice].mm(x)
                return x
    """

    def __init__(self, buffers=None):
        super(BufferDict, self).__init__()
        if buffers is not None:
            self.update(buffers)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, buffer):
        self.register_buffer(key, buffer)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict."""
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its buffer.

        Args:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys."""
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs."""
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values."""
        return self._buffers.values()

    def update(self, buffers):
        r"""Update the :class:`~torch.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`buffers` is an ``OrderedDict``, a :class:`~torch.nn.BufferDict`,
            or an iterable of key-value pairs, the order of new elements in it is
            preserved.

        Args:
            buffers (iterable): a mapping (dictionary) from string to
                :class:`~torch.Tensor`, or an iterable of
                key-value pairs of type (string, :class:`~torch.Tensor`)
        """
        if not isinstance(buffers, container_abcs.Iterable):
            raise TypeError(
                "BuffersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(buffers).__name__
            )

        if isinstance(buffers, container_abcs.Mapping):
            if isinstance(buffers, (OrderedDict, BufferDict)):
                for key, buffer in buffers.items():
                    self[key] = buffer
            else:
                for key, buffer in sorted(buffers.items()):
                    self[key] = buffer
        else:
            for j, p in enumerate(buffers):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "BufferDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def extra_repr(self):
        child_lines = []
        for k, p in self._buffers.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Buffer containing: [{} of size {}{}]".format(
                torch.typename(p), size_str, device_str
            )
            child_lines.append("  (" + k + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError("BufferDict should not be called.")
