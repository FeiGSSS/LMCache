# SPDX-License-Identifier: Apache-2.0
from enum import Enum, auto


class Tier(Enum):
    CPU = auto()
    DISK = auto()
    REMOTE = auto()
