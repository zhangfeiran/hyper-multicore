# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Check the production debug policy against compiler kernel metadata."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.common.mark_utils import arg_mark


_VALIDATOR = (
    Path(__file__).resolve().parents[3]
    / "hyper_multicore/_build/validate_kernel_metadata.py"
)


class TestKernelMetadata(unittest.TestCase):
    """Reject device printf metadata before a native payload is published."""

    @staticmethod
    def _validate(metadata: dict | None) -> subprocess.CompletedProcess[str]:
        """Run the build's validator against one temporary compiler result."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            if metadata is not None:
                (root / "kernel.json").write_text(json.dumps(metadata), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(_VALIDATOR), "--root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_accepts_production_assert_metadata(self) -> None:
        """Feature: preserve compiler assertion support.

        Description: Validate metadata with assertion-only debug information.
        Expectation: Production metadata, including older empty fields, passes.
        """
        for metadata in (
            {},
            {"debugOptions": "", "debugBufSize": 0},
            {"debugOptions": "assert", "debugBufSize": 76800},
            {"kernel": [{"debugOptions": None, "debugBufSize": None}]},
        ):
            with self.subTest(metadata=metadata):
                result = self._validate(metadata)
                self.assertEqual(result.returncode, 0, result.stderr)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_printf_and_oversized_debug_buffers(self) -> None:
        """Feature: prevent production DumpHead output.

        Description: Validate nested printf, malformed fields and large buffers.
        Expectation: Each violation fails and identifies its metadata field.
        """
        for field, value in (
            ("debugOptions", "printf"),
            ("debugOptions", "assert,printf"),
            ("debugOptions", ["assert"]),
            ("debugOptions", True),
            ("debugBufSize", 76801),
            ("debugBufSize", -1),
            ("debugBufSize", "76800"),
            ("debugBufSize", True),
        ):
            with self.subTest(field=field, value=value):
                result = self._validate({"kernels": [{field: value}]})
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"$.kernels[0].{field}", result.stderr)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_missing_metadata(self) -> None:
        """Feature: require compiler metadata.

        Description: Validate a directory without kernel JSON files.
        Expectation: Missing metadata fails instead of publishing a payload.
        """
        result = self._validate(None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("No kernel metadata", result.stderr)
