import setuptools
with open('README.md', 'r') as f:
    long_description = f.read()
setuptools.setup(
    name="sarcasm model",
    version='0.0.1',
    author='DSAN5400 Final Project',
    description='Build Sarcasm Detection Model Class',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_requres={"dev": ["pytest", "flake8", "autopep8"]},
)   