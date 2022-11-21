# DMDO

DMDO is a python implementation for the distributed multidisciplinary design optimization method called nonhierarchical analytical target cascading (NHATC)

For technical and code documentation, please visit [DMDO Webpage](https://ahmed-bayoumy.github.io/DMDO/).

---

**Version 1.3.0**

---

## License & copyright

© Ahmed H. Bayoumy 
---

## Citation

If you use this code, please cite it as shown below.

```pycon
   @software{DMDO_AB,
   author       = {Bayoumy, A.},
   title        = {DMDO},
   year         = 2022,
   publisher    = {Github},
   version      = {1.3.0},
   url          = {https://github.com/Ahmed-Bayoumy/DMDO}
   }
```

## How to use DMDO package

After installing the `DMDO` package using the `pip` command, the functions and classes of `DMDO` module can be imported directly to the python script as follows:

```pycon
from DMDO import *
```

## How to run OMADS from terminal
After installing the libraries listed in the `requirements.txt`, `DMDO/DMDO.py` can be called directly from a 
terminal window under the src directory. The path of the JSON template, which contains the problem input parameters, should be entered as an input argument to the `DMDO.py` call. 

```commandline
python ./src/DMDO/DMDO.py ./tests/test_files/Basic_MDO.yaml
```

After installing the libraries listed in the `requirements.txt`, `DMDO/DMDO.py` can be called directly from a 
terminal window under the main directory. The path of the `YAML` template, which contains the problem input parameters, should be entered as an input argument to the `DMDO.py` call. 





