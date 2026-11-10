# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Isolated arch35 BF16 geometry for the two experimental logical-Cube workers."""
import struct

_RECORD_BYTES = 2016
_A5_FIELDS = {6: 256, 8: 256, 9: 256, 10: 64, 11: 8, 12: 8, 13: 1, 14: 1,
              19: 512*1024, 20: 256*1024, 26: 4, 27: 4, 30: 2, 31: 2, 32: 1}


def retile_a5_bf16(data: bytes, cube_cores: int) -> bytes:
    """Retain tensor/group metadata while using BM256/BN256/BK64 and single L0C.

    Each ping-pong A/B L0 half consumes 32KiB, C consumes 256KiB, and two
    A/B L1 slots consume 512KiB. Only the validated A5 Cube32 layout is exposed.
    The marker and field offsets match the legacy TCubeTiling producer and the
    pinned CANN9.2 kernel_tiling.h; the arch35 worker consumes individual fields.

    Args:
        data: Legacy records produced by the managed MegaMoE tiling tables.
        cube_cores: Logical Cube count; only Ascend 950 with 32 Cubes is supported.

    Returns:
        New records with matrix and routing metadata preserved.

    Raises:
        ValueError: The architecture or legacy record layout is unsupported.
    """
    if cube_cores != 32:
        raise ValueError('A5 BF16 retile requires 32 Cube cores')
    if len(data) != _RECORD_BYTES*cube_cores:
        raise ValueError('A5 BF16 retile requires 2016 bytes per Cube record')
    records = []
    for offset in range(0, len(data), _RECORD_BYTES):
        values = list(struct.unpack('<504I', data[offset:offset+_RECORD_BYTES]))
        markers = [index for index in range(350, len(values)) if values[index] == 393216]
        if len(markers) != 1:
            raise ValueError('Expected one legacy TCubeTiling shared-L1 marker per record')
        start = markers[0]-19
        if start + 50 > len(values):
            raise ValueError('Truncated legacy TCubeTiling record')
        for field, value in _A5_FIELDS.items():
            values[start+field] = value
        records.append(struct.pack('<504I', *values))
    return b''.join(records)


SHARED_BLAZE_TAG = 0x53424631
_SHARED_GROUP_BYTES = 8
_SHARED_VECTOR_BYTES = 80


def retile_a5_shared_bf16(data: bytes, cube_cores: int, *, backward: bool) -> bytes:
    """Retile only the shared GMM records, preserving static shapes and the vector tail.

    Args:
        data: Shared runtime bytes containing one int64 group list, GMM records and an 80-byte vector record.
        cube_cores: Logical Cube count, validated by the existing A5 geometry helper.
        backward: Select four gradient GMM records instead of two forward records.

    Returns:
        Complete shared runtime bytes with unchanged group list and vector tiling.

    Raises:
        ValueError: If the shared runtime does not match the complete static layout.
    """
    if cube_cores != 32:
        raise ValueError("Shared A5 BF16 retile requires 32 Cube cores")
    count = 4 if backward else 2
    stride = _RECORD_BYTES * cube_cores
    if len(data) != _SHARED_GROUP_BYTES + count * stride + _SHARED_VECTOR_BYTES:
        raise ValueError("Shared A5 tiling requires one complete static runtime layout")
    records = [data[:_SHARED_GROUP_BYTES]]
    for offset in range(_SHARED_GROUP_BYTES, _SHARED_GROUP_BYTES + count * stride, stride):
        records.append(retile_a5_bf16(data[offset:offset + stride], cube_cores))
    records.append(data[-_SHARED_VECTOR_BYTES:])
    return b"".join(records)
