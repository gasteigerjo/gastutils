from setuptools import setup, find_packages

install_requires = [
    "matplotlib>=3.3"
]

setup(name='jkutils',
      version='0.1.0',
      description='Joyful & kind utilities',
      url='https://gitlab.lrz.de/klicperajo/jkutils',
      author='Johannes Klicpera',
      author_email='klicpera@in.tum.de',
      packages=find_packages('.'),
      install_requires=install_requires,
      python_requires='>=3.6',
      zip_safe=False)
