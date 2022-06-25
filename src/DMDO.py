from dataclasses import dataclass, field
from abc import ABC
from typing import List
import numpy as np

@dataclass
class subproblem:
    """ Subproblem objects """

@dataclass
class ADMM:
    """ Alternating directions methods of multipliers """

@dataclass
class solver(ABC):
    pass

@dataclass
class variable:
    # Name of the variable
    _name: List[str]
    # Subproblem index
    _SP: int
    # Boolean that indicates if the variable is a CV
    _is_coupling: List[bool]
    # Names of the variables to which the variable is linked
    _links: List[str]
    # Variable dimensions
    _dim: int
    # Lower and upper bound for each components of x
    _lb: List[List[float]]
    _ub: List[List[float]]

    # Dimension of the variable
    _var_dim: int
    # Number of components of X
    _NX: int
    # x_index_to_var_index: For each component x[i] of x, indicate what is the corresponding variable
    _x_index_to_var_index: List[int]
    # % x_index_to_var_subindex: indicate what is the index of x[i] in this variable.
    _x_index_to_var_subindex: List[int]
    # Create a struct to have a fast "name to index" access.
    _name_to_index_struct: dict


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, other: List[str]):
        self._name = other
    def append_name(self, val: str):
        self.name = self.name + [val]
    def extend_name(self, val: str):
        return self.name.extend(val)

    @property
    def SP(self):
        return self._SP
    @SP.setter
    def SP(self, other: int):
        self._SP = other

    @property
    def is_coupling(self):
        return self._is_coupling
    
    @is_coupling.setter
    def is_coupling(self, val: List[bool]):
        self._is_coupling = val

    def append_is_coupling(self, val: bool):
        self.is_coupling = self.is_coupling + [val]

    @property
    def links(self):
        return self._links
    
    @links.setter
    def links(self, other: List[str]):
        self._links = other
    def append_links(self, val: str):
        self.links = self.links + [val]
    def extend_links(self, val: str):
        return self.links.extend(val)
    
    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, other: int):
        self._dim = other
    
    @property
    def lb(self):
        return self._lb
    
    @lb.setter
    def lb(self, other: List[List[float]]):
        self._lb = other

    def append_lb(self, val: List[float]):
        self.lb = self.lb + [val]
    
    @property
    def ub(self):
        return self._ub
    
    @ub.setter
    def ub(self, other: List[List[float]]):
        self._ub = other
    def append_ub(self, val: List[float]):
        self.ub = self.ub + [val]

    @property
    def var_dim(self):
        return self._var_dim
    @var_dim.setter
    def var_dim(self, other: int):
        self._var_dim = other
    
    @property
    def NX(self):
        return self._NX
    @NX.setter
    def NX(self, other: int):
        self._NX = other
    
    @property
    def x_index_to_var_index(self):
        return self._x_index_to_var_index
    @x_index_to_var_index.setter
    def x_index_to_var_index(self, other: List[int]):
        self._x_index_to_var_index = other

    def append_x_index_to_var_index(self, val: int):
        self.x_index_to_var_index = self.x_index_to_var_index + [val]


    @property
    def x_index_to_var_subindex(self):
        return self._x_index_to_var_subindex
    
    @x_index_to_var_subindex.setter
    def x_index_to_var_subindex(self, other: List[int]):
        self._x_index_to_var_subindex = other

    def append_x_index_to_var_subindex(self, val: int):
        self.x_index_to_var_subindex = self.x_index_to_var_subindex + [val]

    @property
    def name_to_index_struct(self):
        return self._name_to_index_struct

    @name_to_index_struct.setter
    def name_to_index_struct(self, other: dict):
        self._name_to_index_struct = other

    def name2varindex(self, name: str) -> int:
        return self.name_to_index_struct[name]

@dataclass
class user_data:
    LAMBDA: float = 0

@dataclass
class problem:
    _is_built: bool = False
    # % The objective function of sub-system index_main is considered as the
    # % general objective function
    _index_main: int = 0
    # % Function to call to perform the subsystem analysis:
    _analysis_file = 'Basic_subsystem_analysis'
    # Number of variables (some of them can be vectors)
    NV: int = 0.0
    var: List[variable] = field(default_factory=list)
    userData: user_data = field(default_factory=user_data)
    XDV_indexes: dict = field(default_factory=dict)

    @property
    def is_built(self):
        return self._is_built
    @is_built.setter
    def is_built(self, other: bool):
        self._is_built = other

    def get_variable(self, *argv):
        n = len(argv)
        out = [] * n
        for k in range(n):
            i = self.var[k].name2varindex(argv[k])
            out.append(self.XDV_indexes[i])
        return out


    def build_problem(self):
        """ """
        # %===========================================
        # % If already built, then leave.
        # %===========================================
        if self.is_built:
            return
        print("\n================ Building problem =============")
        # % Number of variables (some of them can be vectors)
        self.NV = len(self.var)

        for i in range(self.NV):
            v = self.var[i]
            if not isinstance(v.name, str):
                raise(IOError, 'Variable #' + str(i) + ': 1st argument (variable name) must be a char.')
            if v.name == 'auto':
                raise(IOError, 'A variable cannot be named "auto". This is a reserved word')

            v.name_to_index_struct[v.name] = i

            # % Subproblem index
            # % Indicate in which subproblem the variable is involved
            sp_index = v.SP

@dataclass
class options:
    """ Algorithmic options """
    _display: bool = True
    # Updating scheme
    _w_scheme: str = 'median'
    # Initial value for the w vector
    _constraints_cv: bool = True
    # Realistic objective
    _realistic_obj: bool = False
    # Algo stops if this inconsistency is reached
    _inc_stop: float = 1E-12
    # Stop criteria on psize
    _tol: float = 1E-12
    # Algo stops if the inconsistency has not decreased for that many iter
    _no_progress_stop: int = 100
    # Number of Inner/Outer loop iterations
    _ni: int = 100
    _no: int = 100
    # Hyper-parameters of the penalty update scheme
    _beta: float = 2.0
    _gamma: float = 0.5
    # Initial value for the w vector
    _w0: float = 1.0
    # Initial design vector
    _x0: List[float] = field(default_factory=list)
    # Nb of processors
    _nb_proc: float = np.inf
    # Save detail of each subproblems
    _save_subproblem: bool = True
    # Solver
    _solver: str = 'mads'
    _solver_display: bool = False

    # % Mads options
    # %=========================================================
    # Poll size update scheme
    # (How the initial poll size is computed from the final poll size of the previous subsystem optimization)
    _psize: str = 'last'



