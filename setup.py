import os

from setuptools import setup

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = []


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version():
    with open('version.txt') as f:
        return f.read().strip()


setup(
    name='titans',
    version=get_version(),
    packages=['titans'],
    description='A collection of deep learning components built with Colossal-AI',
    long_description=fetch_readme(),
    long_description_content_type='text/markdown',
    license='Apache Software License 2.0',
    url='https://www.colossalai.org',
    project_urls={
        'Forum': 'https://github.com/hpcaitech/Titans/discussions',
        'Bug Tracker': 'https://github.com/hpcaitech/Titans/issues',
        'Examples': 'https://github.com/hpcaitech/ColossalAI-Examples',
        'Documentation': 'http://colossalai.readthedocs.io',
        'Github': 'https://github.com/hpcaitech/Titans',
    },
    install_requires=fetch_requirements('requirements/requirements.txt'),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
)
