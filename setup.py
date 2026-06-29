from setuptools import setup, find_packages

setup(
    name='QLang',
    version='0.1',
    description='Quantum Language Analysis - A library for semantic space analysis and contextual contours',
    author='Alejandro Martinez-Mingo',
    author_email='contact@example.com',
    url='https://github.com/alejandrommingo/QLang',
    download_url='https://github.com/alejandrommingo/QLang/archive/v0.1.tar.gz',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        'matplotlib',
        'fuzzywuzzy',
        'pytest',
        'tqdm',
        'python-Levenshtein',
        'scikit-learn',
        'safetensors'
    ],
    extras_require={
        'bert': ['transformers', 'torch', 'wordfreq'],
        'lsa': ['gensim']
    },
    include_package_data=True,  # Asegúrate de incluir esto
    package_data={
        'QLang': ['resources/*']  # Incluye todos los archivos en la carpeta resources
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    python_requires='>=3.8',
    keywords='nlp, quantum, language, semantics, transformers, bert, gpt2'
    # Más configuraciones como author, license, etc.
)
