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
"""Validate device UB discovery for CANN releases without the platform API."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hyper_multicore import _device


class TestLegacyDeviceUb(unittest.TestCase):
    """Use only the current device's exact installed platform metadata."""

    def tearDown(self) -> None:
        _device._platform_query.cache_clear()
        _device.device_ub_bytes.cache_clear()

    def test_cann_85_reads_device_specific_config_and_rejects_missing_metadata(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "opp").mkdir()
            (root / "opp/version.info").write_text("Version=8.5.2\n", encoding="utf-8")
            config = root / "x86_64-linux/data/platform_config"
            config.mkdir(parents=True)
            (config / "Ascend910B2C.ini").write_text("[AICoreSpec]\nub_size=196608\n", encoding="utf-8")
            npu = SimpleNamespace(get_device_name=Mock(return_value="Ascend910B2C"))
            with patch.dict("os.environ", {"ASCEND_HOME_PATH": temporary}), \
                    patch.object(_device.ctypes, "CDLL", return_value=SimpleNamespace()), \
                    patch.object(_device.platform, "machine", return_value="x86_64"), \
                    patch.object(_device.torch, "npu", npu, create=True):
                self.assertEqual(_device.device_ub_bytes(3), 196608)
                npu.get_device_name.assert_called_once_with(3)
                for name in ("Ascend910_9382", "Ascend910A", "../Ascend910B2C"):
                    _device.device_ub_bytes.cache_clear()
                    npu.get_device_name.return_value = name
                    self.assertEqual(_device.device_ub_bytes(3), 0)
                _device.device_ub_bytes.cache_clear()
                npu.get_device_name.return_value = "Ascend910B2C"
                (config / "Ascend910B2C.ini").write_text("[AICoreSpec]\nub_size=invalid\n", encoding="utf-8")
                self.assertEqual(_device.device_ub_bytes(3), 0)

    def test_new_cann_still_requires_platform_api(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "opp").mkdir()
            (root / "opp/version.info").write_text("Version=9.1.0\n", encoding="utf-8")
            with patch.dict("os.environ", {"ASCEND_HOME_PATH": temporary}), \
                    patch.object(_device.ctypes, "CDLL", return_value=SimpleNamespace()):
                with self.assertRaisesRegex(RuntimeError, "platform-query API"):
                    _device.device_ub_bytes(0)


if __name__ == "__main__":
    unittest.main()
