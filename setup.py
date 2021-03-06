from Cython.Distutils import build_ext
import numpy as np
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

genome_module = Extension(
    "skylar.sequences._sequence",
    ["skylar/sequences/_sequence.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "skylar.targets._genomic_features",
    ["skylar/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

setup(name="skylar",
      version="1.0",
      long_description=None,
      long_description_content_type='text/markdown',
      description="SeleneSDK migrated from PyTorch to TensorFlow",
      packages=find_packages(),
      url="https://github.com/fulcrum1378/tf_selene",
      package_data={
          "skylar.interpret": [
              "data/gencode_v28_hg38/*",
              "data/gencode_v28_hg19/*"
          ],
          "skylar.sequences": [
              "data/*"
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD Liceense",
          "Topic :: Scientific/Engineering :: Bio-Informatics"
      ],
      ext_modules=[genome_module, genomic_features_module],
      cmdclass={'build_ext': build_ext},
      install_requires=[
          "cython>=0.27.3",
          'click',
          "h5py",
          "matplotlib>=2.2.3",
          "numpy",
          "pandas",
          "plotly",
          "pyfaidx",
          "pytabix",
          "pyyaml>=5.1",
          "scikit-learn",
          "scipy",
          "seaborn",
          "six",
          "statsmodels",
          "tensorflow",
      ],
      entry_points={
          'console_scripts': [
              'skylar = skylar.cli:main',
          ],
      },
      )
