from setuptools import setup, find_packages

setup(
    name="LPRNet",
    version="0.0.1",
    author="Heonjin Kwon",
    author_email="kwon@4ind.co.kr",
    description="A license plate recognition Module written in pytorch-lightning",
    keywords=['pytorch', 'pytorch-lightning', 'license-plate-recognition'],
    install_requires=[
        'pytorch-lightning>=1.7.0, <=1.9.0',
        'numpy>=1.17.1',
        'tqdm>=4.57.0',
        'PyYAML>=5.4',
        'imutils>=0.4.0',
        'rich>=10.2.2',
        'albumentations>=1.3.0'
    ],
    packages=find_packages(),
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',   
)
