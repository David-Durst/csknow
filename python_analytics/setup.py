from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='csknow-python-analytics',
    version='0.0.1',
    url='https://github.com/David-Durst/csknow/python_analytics',
    license='MIT',
    maintainer='David Durst',
    maintainer_email='davidbdurst@gmail.com',
    description='CSKnow Python Analytics',
    packages=[
        "csknow-python-analytics",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pillow",
        "psycopg2-binary",
        "opencv-python",
        "pytesseract",
        "pyautogui",
        "pydirectinput",
        "keyboard"
    ],
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
