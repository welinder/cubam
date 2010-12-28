from distutils.core import setup, Extension

# C++ implementation sources
cppDir = 'src'
cppFiles = [
  'Binary1dSignalModel.cpp',
  'BinaryModel.cpp',
  'BinaryNdSignalModel.cpp',
  'BinarySignalModel.cpp',
  'Model.cpp',
  'annmodel.cpp',
  'utils.cpp'
]
sources = ['%s/%s' % (cppDir, fn) for fn in cppFiles]

setup (name='CUBAM',
  version='1.0',
  description='Python implementation of the Caltech-UCSD Binary Annotation Model',
  author = 'Peter Welinder',
  author_email = 'peter@welinder.se',
  url = 'http://github.com/welinder/cubam',
  ext_modules = [Extension('cubamcpp', sources=sources)],
  packages=['cubam'])