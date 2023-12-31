from setuptools import setup, find_packages

VERSION = '0.0.4' 
DESCRIPTION = 'Toolkits for text processing and augmentation for Bangla NLP'

setup(
        name="banglanlptoolkit", 
        version=VERSION,
        author="A F M Mahfuzul Kabir",
        author_email="<afmmahfuzulkabir@gmail.com>",
        description=DESCRIPTION,
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=[
            'bnunicodenormalizer',
            'numpy',
            'pandas',
            'sentencepiece',
            'torch',
            'tqdm',
            'transformers',
            'langdetect',
        ],
        dependency_links=['http://github.com/csebuetnlp/normalizer.git'],
        url='https://github.com/Kabir5296/banglanlptoolkit',
        project_urls={'Repository': 'https://github.com/Kabir5296/banglanlptoolkit'},
)