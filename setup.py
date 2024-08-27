"""biogen package installation metadata."""

# Currently unused.

from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "biogen.version",
    str(Path(__file__).parent / "src" / "biogen" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="biogen",
    description="Synthetic data generation for biomedical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version=version.__version__,
    license="GPL-3.0",
    author="pedro0sorio",
    author_email="pedro.c.osorio@gmail.com",
    url="https://github.com/bayer-int/MEP-LDM",
    keywords=["Machine learning", "artificial intelligence"],
    test_suite="tests",
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    extras_require={},
    install_requires=[
        "torchvision",
        "pytorch_lightning",
        "transformers",
        "diffusers>=0.11.0",
        "umap-learn",
        "networkx==2.8.6",
        "matplotlib",
        "panel",
        "einops",
        "torch==2.1.2",
        "datasets==2.16.1",
        "bitsandbytes==0.42.0",
        "xformers==0.0.23.post1",
        "ray==2.32.0",
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GPL-3.0 License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
    ],
    entry_points="""
            [console_scripts]
            biogen=biogen.cli:cli
        """,
)
