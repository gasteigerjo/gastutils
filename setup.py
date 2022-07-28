from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib>=3.3",
    "seaborn",
    "networkx",
]

setup(name='gastutils',
      version='0.1.0',
      description='Gasteiger utilities',
      url='https://github.com/gasteigerjo/gastutils',
      author='Johannes Gasteiger',
      author_email='gasteiger@mailbox.org',
      packages=find_packages('.'),
      install_requires=install_requires,
      python_requires='>=3.6',
      zip_safe=False)
