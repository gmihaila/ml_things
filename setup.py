"""
Heavily inspired from https://github.com/huggingface/transformers

Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.

2. Unpin specific versions from setup.py (like isort).

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
   
   Usualy run: "python setup.py sdist bdist_wheel"

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

8. Add the release version to docs/source/_static/js/custom.js and .circleci/deploy.sh

9. Update README.md to redirect to correct documentation.
"""

from setuptools import setup, find_packages

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
        "scikit-learn",
        "numpy",
        "requests",
        "tqdm >= 4.27",
        "ftfy >= 5.8",
        "matplotlib>=3.4.0",
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
