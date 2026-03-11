# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import MutableMapping
from typing import Generic, Optional, TypeVar
import abc

KeyType = TypeVar("KeyType")
MapType = TypeVar("MapType", bound=MutableMapping)


class BaseCachePolicy(Generic[KeyType, MapType], metaclass=abc.ABCMeta):
    """
    Interface for cache policy.
    """

    @abc.abstractmethod
    def init_mutable_mapping(self) -> MapType:
        """
        Initialize a mutable mapping for cache storage.

        Return:
            A mutable mapping that can be used to store cache entries.
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: MapType,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is used

        Input:
            key: an object of KeyType
            cache_dict: a dict consists of current cache
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        """
        Update cache_dict and internal states when a cache is stored

        Input:
            key: an object of KeyType
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        """
        Update internal states when a cache is force evicted

        Input:
            key: an object of KeyType
        """
        raise NotImplementedError

    # TODO(Jiayi): we need to unify the `Any` type in the `MutableMapping`
    @abc.abstractmethod
    def get_evict_candidates(
        self,
        cache_dict: MapType,
        num_candidates: int = 1,
    ) -> list[KeyType]:
        """
        Evict cache when a new cache comes and the storage is full

        Input:
            cache_dict: a dict consists of current cache
            num_candidates: number of candidates to be evicted

        Return:
            return a list of keys to be evicted
        """
        raise NotImplementedError

    def record_context(
        self,
        key: KeyType,
        *,
        prefix_pos: Optional[int] = None,
    ) -> None:
        """
        Record optional context for the next policy update of ``key``.

        Args:
            key: Cache key being updated.
            prefix_pos: Optional chunk position within the current request.
        """

    def periodic_maintenance(
        self,
        cache_dict: MapType,
    ) -> None:
        """
        Run periodic policy maintenance.

        This hook is invoked by storage backends from a background task when
        a policy needs aging or periodic reindexing. Policies that do not need
        maintenance should keep the default no-op implementation.

        Args:
            cache_dict: The current cache mapping for the backend.
        """

    def requires_periodic_maintenance(self) -> bool:
        """
        Report whether this policy needs background maintenance.

        Returns:
            True when the backend should periodically call
            :meth:`periodic_maintenance`, otherwise False.
        """
        return False
