from setuptools import setup, find_packages

#with open("README.md", "r") as readme_file:
#    readme = readme_file.read()

#requirements = []
#
#setup(
#    name="ml_things",
#    version="0.0.1",
#    author="George Mihaila",
#    author_email="georgemihaila@my.unt.edu",
#    description="A package to keep all helping functions used in Machine Learning together",
#    long_description=open("README.md", "r", encoding="utf-8").read(),
#    long_description_content_type="text/markdown",
#    url="https://github.com/gmihaila/ml_things",
#    packages=find_packages(),
#    install_requires=requirements,
#    classifiers=[
#        "Programming Language :: Python :: 3.7",
#        "License :: MIT License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
#    ],
#)

extras = {}

setup(
    name="ml_things",
    version="0.0.1",
    author="George Mihaila",
    author_email="georgemihaila@my.unt.edu",
    description="Useful functions when working with Machine Learning in Python",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning pytorch tensorflow numpy",
    license="Apache",
    url="https://github.com/gmihaila/ml_things",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm >= 4.27",
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
