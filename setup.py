from setuptools import setup, find_packages

VERSION = '1.1.7' 
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
        
        classifiers = [
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8"
        ],
        
        keywords=['Bangla','NLP','toolkit','punctuation','augmentation','normalizer'],
        
        url='https://github.com/Kabir5296/banglanlptoolkit',
)