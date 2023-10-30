import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="TopoReg_QSAR",
    version="0.0.1",
    author="Daniel Nolte and Ruibo Zhang",
    author_email="daniel.nolte@ttu.edu",
    description=
    "Repository containing code and sample data to recreate results from the paper Topological Regression in Quantitative Structure-Activity Relationship Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ribosome25/TopoReg_QSAR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[ 'pandas', 'numpy', 'scipy', 'scikit-learn', 'networkx'],
    python_requires='>=3.8'
    )