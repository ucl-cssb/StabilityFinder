from setuptools import setup, find_packages

setup(name='stability-checker',
      version='1.2.0.dev1',
      description='Checking the stability of a model under parameter uncertainty',
      author='Miriam Leon',
      author_email='miriam.leon.12@ucl.ac.uk',
      url='https://github.com/Mirelio/StabilityChecker.git',
      install_requires=['numpy',
                'math',
                'scipy',
                'logging',
                'copy',
                'operator',
                'time',
                'sys',
                'cudasim']
      )