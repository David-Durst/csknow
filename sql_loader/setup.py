from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='csknow_sql_loader',
    version='0.0.1',
    url='https://github.com/David-Durst/csknow/sql_loader',
    license='MIT',
    maintainer='David Durst',
    maintainer_email='davidbdurst@gmail.com',
    description='Load CSKnow csvs into CSGO',
    packages=[
        "csknow_sql_loader",
    ],
    install_requires=[
    ],
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
