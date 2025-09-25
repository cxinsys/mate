from setuptools import setup, find_packages

setup(name='mate-cxinsys',
      version='0.1.22',
      description='MATE',
      url='https://github.com/cxinsys/mate',
      author='Complex Intelligent Systems Laboratory (CISLAB)',
      author_email='daewon4you@gmail.com',
      license='BSD-3-Clause',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      package_data={
            'mate-cxinsys': ['mate/transferentropy/infodynamics.jar']
      },
      include_package_data=True,
      install_requires=['numpy', 'scipy', 'lightning', 'JPype1', 'scikit-learn'],
      zip_safe=False,)
