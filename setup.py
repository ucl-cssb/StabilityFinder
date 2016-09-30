from distutils.core import setup

setup(name='stabilityfinder',
      version='1.2.0.dev1',
      description='Finding the stability of a model under parameter uncertainty',
      author='Miriam Leon',
      author_email='miriam.leon.12@ucl.ac.uk',
      url='https://github.com/ucl-cssb/StabilityFinder.git',
      scripts=['stabilityfinder/stabilityfinder_scr'],
      packages=['stabilityfinder'],
      package_data={
          '':['*.R']
      }
     )
