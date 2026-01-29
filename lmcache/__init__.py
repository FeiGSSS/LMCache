
# SPDX-License-Identifier: Apache-2.0

# NOTE: Import torch early so its shared libraries (e.g. libc10.so) are loaded
# before importing the C++/CUDA extension module `lmcache.c_ops`.
import torch as _torch  # noqa: F401

