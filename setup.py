from setuptools import setup, find_packages

VERSION = '1.1.8' 
DESCRIPTION = 'Toolkits for text processing and augmentation for Bangla NLP'

REQUIREMENTS = [
    'transformers==4.42.4',
    'torch==2.3.1',
    'bnunicodenormalizer==0.1.7',
    'sentencepiece==0.2.0',
    'langdetect==1.0.9',
    'pandarallel',
    'pqdm',
#     'normalizer @ git+https://github.com/csebuetnlp/normalizer'
]

setup(
        name="banglanlptoolkit", 
        version=VERSION,
        
        author="A F M Mahfuzul Kabir",
        author_email="<afmmahfuzulkabir@gmail.com>",
        description=DESCRIPTION,
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        
        classifiers = [
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8"
        ],
        
        keywords=['Bangla','NLP','toolkit','punctuation','augmentation','normalizer', 'tokenize'],
        
        url='https://github.com/Kabir5296/banglanlptoolkit',
)
