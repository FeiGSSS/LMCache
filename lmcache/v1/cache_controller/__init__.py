# SPDX-License-Identifier: Apache-2.0
__all__ = [
    "LMCacheClusterExecutor",
    "LMCacheWorker",
]


def __getattr__(name: str):
    if name == "LMCacheClusterExecutor":
        # First Party
        from lmcache.v1.cache_controller.executor import LMCacheClusterExecutor

        return LMCacheClusterExecutor
    if name == "LMCacheWorker":
        # First Party
        from lmcache.v1.cache_controller.worker import LMCacheWorker

        return LMCacheWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
