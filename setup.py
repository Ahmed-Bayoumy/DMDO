from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="DMDO",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2401',
    packages=find_packages(include=['DMDO', 'DMDO.*']),
    description="Distributed Multidisciplinary Design Optimization (DMDO)",
    install_requires=[
      'pandas',
      'NOBM',
      'numpy',
      'matplot',
      'OMADS',
      'genericpath',
      'math',
      'enum'
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
  )
