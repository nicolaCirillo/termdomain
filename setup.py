from setuptools import setup, find_packages


setup(
      name='termdomain',
      version='0.1',
      description='A term extraction tool based on word embeddings.',
      url='#',
      author='Nicola Cirillo',
      author_email='nicola.cirillo96@outlook.it',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
              'numpy',
              'pandas',
              'nltk',
              'conllu',
              'spacy',
              ],
      )

