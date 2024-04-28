from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="DMDO",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2404',
    packages=find_packages(include=['DMDO', 'DMDO.*']),
    description="Distributed Multidisciplinary Design Optimization (DMDO)",
    install_requires=[
      'pandas==2.2.2',
      'NOBM==2404.1',
      'numpy==1.23.2',
      'OMADS==2404.1',
      'pyyaml'
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
