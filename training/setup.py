from setuptools import find_packages
from setuptools import setup

# GPU doesnt work with tensorflow 2.0. https://issuetracker.google.com/issues/144390939
REQUIRED_PACKAGES = ["tensorflow-transform"] #TODO: Debug training fails if this is empty

setup(
    name="trainer",
    version="2.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Name Gender Prediction Package"
)

