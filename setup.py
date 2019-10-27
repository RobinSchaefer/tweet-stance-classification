import codecs
from setuptools import find_packages, etup

with codecs.open('README.md', mode='r', encoding='utf-8') as f_in:
    readme = f_in.read()

with codecs.open('requirements.txt', mode='r', encoding='utf-8') as f_in:
    install_requires = f_in.read()

setup(
    name='TweetStanceClassification',
    version='1.0',
    author='Robin Schaefer',
    author_email='robin.schaefer@uni-potsdam.de',
    url='https://github.com/RobinSchaefer/tweet-stance-classification',
    description='Stance classification for English tweets',
    long_description=readme,
    packages=find_packages(),
    install_requires=install_requires,
)
