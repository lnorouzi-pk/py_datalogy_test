from setuptools import setup

setup(name='py_datalogy',
      version='0.1',
      description='A library to investigate the data and provide necessary EDA information.',
      url='https://github.com/lnorouzi-pk/py_datalogy',
      author='Leila Norouzi',
      author_email='lnorouzi@prokarma.com',
      license='MIT',
      packages=['py_datalogy'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'python-dateutil'
      ],
      classifiers=[
        'Development Status :: 1 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)