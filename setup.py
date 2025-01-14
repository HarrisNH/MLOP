from setuptools import find_packages, setup

setup(
    name="Packages_finder",
    version="0.1.0",
    description="A short description of the project.",
    author="Your name (or your organization/company/team)",
    packages=find_packages(where="MNIST_corrupt"),
    license="MIT",
)
