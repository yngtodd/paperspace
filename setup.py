from setuptools import setup, find_packages


setup(
    name='paperspace',
    version="0.0.1",
    packages=find_packages(),
    install_requires=['numpy', 'scikit-optimize', 'scikit-learn', 'mpi4py'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Experiments and Results for HyperSpace",
    license="MIT",
    keywords="parallel Bayesian optimization smbo",
    url="https://github.com/yngtodd/paperspace",
)
