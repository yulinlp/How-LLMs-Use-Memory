from setuptools import setup, find_packages

setup(
    name="steem_adapt",
    version="0.1.0",
    packages=find_packages(include=['steem_adapt', 'steem_adapt.*']),
    python_requires=">=3.10",
)
