from setuptools import setup, find_packages

about = {}
with open('txtanot/version.py', 'r') as f:
    exec(f.read(), about)

setup(
    name=about['__name__'],
    version=about['__version__'],
    description=about['__description__'],
    author=about['__author__'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[

    ],
    entry_points={

    }
)