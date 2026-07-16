from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys


def test_python2_smoke_runs_without_ros():
    script = os.path.join(os.path.dirname(__file__), "python2_smoke.py")
    output = subprocess.check_output([sys.executable, script], stderr=subprocess.STDOUT)
    if not isinstance(output, str):
        output = output.decode("utf-8")
    assert output.startswith("OK: Python 2/3 protocol")
