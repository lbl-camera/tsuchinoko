from os import path
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 9)
if sys.version_info < min_version:
    error = """
tsuchinoko does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='tsuchinoko',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="An adaptive optics alignment tool for ALS beamlines utilizing gpCAM.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Ronald J Pandolfi",
    author_email='ronpandolfi@lbl.gov',
    url='https://github.com/lbl-camera/tsuchinoko',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests', 'examples']),
    entry_points={
        "gui_scripts": [
            'tsuchinoko = tsuchinoko:launch_client',
        ],
        "console_scripts": [
            'tsuchinoko_demo = tsuchinoko:launch_server',
            'tsuchinoko_bootstrap = tsuchinoko:bootstrap'
        ],
    },
    include_package_data=True,
    package_data={
        'tsuchinoko': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    license='GPLv3+',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    extras_require={
        "dev": ["pyinstaller"],
        "docs": ["sphinx", "sphinx-markdown-tables", "numpydoc", "sphinx_copybutton", "myst_parser", "sphinx_rtd_theme", "sphinx_rtd_dark_mode"],
        "tests": ["pytest", "coverage", "coveralls", "codecov", "pylint", "pytest-qt", "pytest-cov", "pytest-lazy-fixture"],
    },
)
