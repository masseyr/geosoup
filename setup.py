import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geosoup",
    version="0.1.9",
    author="Richard Massey",
    author_email="rm885@nau.edu",
    description="Geospatial data manipulation using GDAL in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masseyr/geosoup",
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    install_requires=[
        'psutil',
        'h5py',
        'numpy',
        'scikit-learn',
        'scipy',
    ],
    keywords='geospatial raster vector global spatial regression hierarchical samples random',
)
