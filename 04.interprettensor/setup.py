from setuptools import setup
from setuptools import find_packages

exec(open('c:/Users/bueno/Documents/Deeplearning/04.interprettensor/_version.py').read())
setup(
    name='lrptoolbox',
    packages=['lrptoolbox'],
    version=__version__,
    description='A tensorflow wrapper with LRP implementations ',
    author=' Vignesh Srinivasan, Sebastian Lapuschkin, Gregoire Montavon',
    author_email='vignesh.srinivasan@hhi.fraunhofer.de',
    url='https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/tf/tensorflow',
    download_url='https://github.com/sebastian-lapuschkin/lrp_toolbox/tree/tf/tensorflow/tarball/' + __version__,
    license='MIT',
    install_requires=['tensorflow>=1.0.0'],
    keywords=['tensorflow', 'LRP', 'wrapper', 'slim', 'toolbox'],
)
