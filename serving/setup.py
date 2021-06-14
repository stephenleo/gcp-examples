from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name="name_gender_predictor",
    version="2.0",
    scripts=["predictor.py"],
    install_requires=REQUIRED_PACKAGES,
    description="Gender Prediction Serving Package"
)