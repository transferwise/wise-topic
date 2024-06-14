from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name="wise-topic",
    version="0.0.1",
    description="LLM-only topic extraction and classification",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Wise",
    url='https://github.com/transferwise/wise-topic',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    install_requires=[
        "langchain",
        "langchain_core",
        "langchain_openai",
        "numpy",
        "scikit_learn"
    ],
    extras_require={
        "test": [
            "flake8",
            "pytest",
            "pytest-cov"
        ],
    },
    packages=find_packages(
        include=[
            'wise_topic',
            'wise_topic.*'
        ],
        exclude=['tests*'],
    ),
    include_package_data=True,
    keywords='wise-topic',
)
