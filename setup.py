import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geosoup",
    version="0.1.65",
    author="Richard Massey",
    author_email="rm885@nau.edu",
    license='Apache License 2.0',
    description="Geospatial data manipulation using GDAL in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masseyr/geosoup",
    packages=setuptools.find_packages(),
    classifiers=[
        'Topic :: Scientific/Engineering :: GIS',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'psutil',
        'numpy',
    ],
    keywords='geospatial raster vector global spatial regression hierarchical samples random',
)
