from setuptools import setup, find_packages

setup(name='mate-cxinsys',
      version='{{VERSION_PLACEHOLDER}}',
      description='MATE',
      url='https://github.com/cxinsys/mate',
      author='Complex Intelligent Systems Laboratory (CISLAB)',
      author_email='daewon4you@gmail.com',
      license='BSD-3-Clause',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'lightning'],
      zip_safe=False,)
