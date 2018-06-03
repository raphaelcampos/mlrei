#!/usr/bin/env python

from distutils.core import setup
print(list(open('requirements.txt').readlines()))
setup(name='Machine Learning Reimplemented',
      version='0.0.0',
      description='''Implementation of machine learning methods and papers experiments in a didatic fashion,
                     in order to have a better understanding of how the methods work.''',
      author='Raphael Campos',
      author_email='raphaelrcampos@yahoo.com.br',
      install_requires=list(open('requirements.txt').readlines())
)