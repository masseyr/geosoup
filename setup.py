import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geosoup",
    version="0.1.1",
    author="Richard Massey",
    author_email="rm885@nau.edu",
    description="Geospatial data manipulation using GDAL in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masseyr/geosoup",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'psutil>=5.6.2',
        'h5py>=2.10.0',
        'numpy>=1.16.5',
        'earthengine-api>=0.1.175',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.2',
        'GDAL>=2.2.3',
    ]
)
