from setuptools import setup, find_packages

setup(
    name='banglanlptoolkit',
    version='0.0.2',
    author='A F M Mahfuzul Kabir',
    author_email='afmmahfuzulkabir@gmail.com',
    description='Toolkits for text processing and augmentation for Bangla NLP',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Kabir5296/banglanlptoolkit',
    project_urls={
        'Repository': 'https://github.com/Kabir5296/banglanlptoolkit'
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    package_dir={'': 'banglanlptoolkit'},
    packages=find_packages(where='banglanlptoolkit'),
    python_requires='>=3.8',
    install_requires=[
        'bnunicodenormalizer',
        # 'normalizer',
        # 'git@github.com/csebuetnlp/normalizer.git@main#egg=normalizer',
        'numpy',
        'pandas',
        'sentencepiece',
        'torch',
        'tqdm',
        'transformers'
    ],
    dependency_links=['http://github.com/csebuetnlp/normalizer.git']
)

# [metadata]
# name = banglanlptoolkit
# version = 1.0.7
# author = A F M Mahfuzul Kabir
# author_email = afmmahfuzulkabir@gmail.com
# description = Toolkits for text processing and augmentation for Bangla NLP
# long_description = file: README.md
# long_description_content_type = text/markdown
# url = https://github.com/Kabir5296/banglanlptoolkit
# project_urls =
#     repository = https://github.com/Kabir5296/banglanlptoolkit
# classifiers =
#     Programming Language :: Python :: 3
#     License :: OSI Approved :: MIT License
#     Operating System :: OS Independent

# [options]
# package_dir =
#     = banglanlptoolkit
# packages = find:
# python_requires = >=3.8

# install_requires = 
#     bnunicodenormalizer
#     # normalizer @ git+https://github.com/csebuetnlp/normalizer@main
#     numpy
#     pandas
#     sentencepiece
#     torch
#     torchaudio
#     torchvision
#     tqdm
#     transformers



# [options.packages.find]
# where = banglanlptoolkit