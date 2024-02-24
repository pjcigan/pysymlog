import setuptools

with open('README.md','r') as fh:
    long_description=fh.read()

setuptools.setup(
    name='pysymlog',
    version='1.0.1',
    url='https://github.com/pjcigan/pysymlog', #'http://pysymlog.readthedocs.io',
    license='MIT',
    author='Phil Cigan',
    author_email='',
    description='Symmetric (signed) logarithmic scale',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    py_modules=['pysymlog',],
    #python_requires='>2.7',
    install_requires=['numpy',], 
    extras_require={'matplotlib':  ["matplotlib",], 'plotly':["plotly",] },
)
