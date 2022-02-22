from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='learn_bot',
    version='0.0.1',
    url='https://github.com/David-Durst/csknow/learn_bot',
    license='MIT',
    maintainer='David Durst',
    maintainer_email='davidbdurst@gmail.com',
    description='CSKnow Learning A Bot',
    packages=[
        "learn_bot",
    ],
    install_requires=[
    ],
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
