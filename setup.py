from setuptools import setup

setup(
    name="tools",
    version='0.0.1',
    packages=["tools"],
    install_requires=['pandas', 'numpy', 'requests', 'setuptools', 'seaborn', 'sklearn'],
    scripts = ['scripts/generate_pkg_template']
    )
