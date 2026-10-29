"""Metadata is in pyproject.toml; this only ships skills/sweepx in the wheel."""
import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class build_py_with_skill(build_py):
    """Copy skills/sweepx into the wheel as sweepx/_skill/sweepx."""

    def run(self):
        super().run()
        src = os.path.join(ROOT_DIR, "skills", "sweepx")
        dest = os.path.join(self.build_lib, "sweepx", "_skill", "sweepx")
        if not os.path.isfile(os.path.join(src, "SKILL.md")):
            raise RuntimeError(f"{src} is missing: the wheel must ship the skill")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


setup(cmdclass={"build_py": build_py_with_skill})
