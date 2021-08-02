
from setuptools import setup

setup(
    name="utils",
    version='0.0.1',
    packages=["utils"],
    install_requires=['pandas', 'numpy', 'requests', 'setuptools'],
    scripts = ['scripts/generate_pkg_template']
    )
