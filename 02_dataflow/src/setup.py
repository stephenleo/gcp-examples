import setuptools

required_packages = ["fasttext"]

setuptools.setup(
    name="DataflowEmbeddingGen",
    version="0.1",
    author="stephenleo",
    author_email="stephen.leo87@gmail.com",
    description="Dataflow Apache Beam code to generate embeddings",
    install_requires=required_packages
)