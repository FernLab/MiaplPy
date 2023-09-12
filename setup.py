#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('docs/README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

version = {}
with open("miaplpy/version.py") as version_file:
    exec(version_file.read(), version)

req = ["cython", "numpy", "pyproj", "matplotlib", "numba", "scipy", "mintpy", "Pillow", "h5py", "overpy", "miaplpy", "gstools", "networkx"]

req_setup = ['pytest-runner']

req_test = ['pytest>=3', 'pytest-cov', 'pytest-reporter-html1', 'urlchecker']

req_doc = [
    'sphinx>=4.1.1',
    'sphinx-argparse',
    'sphinx-autodoc-typehints',
    'sphinx_rtd_theme'
]

req_lint = ['flake8', 'pycodestyle', 'pydocstyle']

req_dev = ['twine'] + req_setup + req_test + req_doc + req_lint

setup(
    author="",
    author_email='',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'None',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="MiaplPy",
    entry_points={
        'console_scripts': [
            'miaplpyApp=miaplpy.miaplpyApp:main',
            'check_ifgs.py=miaplpy.check_ifgs:main',
            'correct_geolocation.py=miaplpy.correct_geolocation:main',
            'cpxview.py=miaplpy.cpxview:main',
            'generate_ifgram.py=miaplpy.generate_ifgram:main',
            'generate_temporal_coherence.py=miaplpy.generate_temporal_coherence:main',
            'generate_unwrap_mask.py=miaplpy.generate_unwrap_mask:main',
            'load_ifgram.py=miaplpy.load_ifgram:main',
            'load_slc_geometry.py=miaplpy.load_slc_geometry:main',
            'network_inversion.py=miaplpy.network_inversion:main',
            'phase_linking.py=miaplpy.phase_linking:main',
            'prep_slc_isce.py=miaplpy.prep_slc_isce:main'
        ],
    },
    extras_require={
        "doc": req_doc,
        "test": req_test,
        "lint": req_lint,
        "dev": req_dev
    },
    install_requires=req,
    license="None",
    keywords='miaplpy',
    long_description=readme,
    name='miaplpy',
    packages=find_packages(include=['miaplpy', 'miaplpy.*']),
    package_data={"miaplpy": ["defaults/*"]},
    include_package_data=True,
    setup_requires=req_setup,
    test_suite='tests',
    tests_require=req_test,
    url='https://github.com/FernLab/MiaplPy',
    version=version['__version__'],
    zip_safe=False,
)
