from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="ml_things",
    version="0.0.1",
    author="George Mihaila",
    author_email="georgemihaila@my.unt.edu",
    description="A package to keep all helping functions used in Machine Learning together",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/gmihaila/ml_things",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
