import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geosoup",
    version="0.1.0",
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
    python_requires='>=3.7',
)
