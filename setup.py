from setuptools import setup, find_packages


# Function to read the requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()


setup(
    name="TezzChain",
    version="0.1.1",
    author="Japkeerat Singh",
    author_email="japkeerat21@gmail.com",
    description="Low Code RAG building tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/japkeerat/TezzChain",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "ollama==0.4.4",
        "chromadb==0.6.2",
        "unstructured==0.16.12",
        "sqlalchemy==2.0.37",
        "psycopg2-binary==2.9.10",
        "PyYAML==6.0.2",
    ],
    include_package_data=True,
)
