# ------------------------------------------------------------------------------------#
#  Multidisciplinary Design Optimization - DMDO                                      #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#

import copy
import csv
import enum
from genericpath import isfile
import json
from multiprocessing.dummy import Process
from multiprocessing.sharedctypes import Value

from dataclasses import dataclass, field
from tkinter.font import NORMAL
from typing import List, Dict, Any, Callable, Protocol, Optional

import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import math

import OMADS
from enum import Enum, auto

@dataclass
class BMMDO:
	""" Simple testing problems """

@dataclass
class double_precision:
		decimals: int
		""" Decimal precision control """

		def truncate(self, value: float) -> float:
				if abs(value) != inf:
						multiplier = 10 ** self.decimals
						return int(value * multiplier) / multiplier if value != inf else inf
				else:
						return value

class VAR_TYPE(Enum):
	CONTINUOUS = auto()
	INTEGER = auto()
	BINARY = auto()
	CATEGORICAL = auto()
	ORDINAL = auto()

class VALIDATOR(Enum):
	PRE = auto()
	RUNNING = auto()
	POST = auto()

class BARRIER_TYPE(Enum):
	EXTREME = auto()
	PROGRESSIVE = auto()
	FILTER = auto()

class PSIZE_UPDATE(Enum):
	DEFAULT = auto()
	SUCCESS = auto()
	MAX = auto()
	LAST = auto()

class w_scheme(Enum):
	MEDIAN = auto()
	MAX = auto()
	NORMAL = auto()
	RANK = auto()
	
class MODEL_TYPE(Enum):
	SURROGATE = auto()
	DATA = auto()
	SIMULATION = auto()
	NEUTRAL = auto()

class COUPLING_TYPE(Enum):
	SHARED = auto()
	FEEDBACK = auto()
	FEEDFORWARD = auto()
	UNCOUPLED = auto()

class COUPLING_STRENGTH(Enum):
	TIGHT = auto()
	LOOSE = auto()

class MDO_ARCHITECTURE(Enum):
	MDF = auto()
	IDF = auto()

@dataclass
class variableData:
	index: int
	name: str
	sp_index: int
	coupling_type: int
	link: str
	dim: int
	value: float
	baseline: float
	scaling: float
	lb: float = None
	ub: float = None
	type: int = VAR_TYPE.CONTINUOUS

	def __sub__(self, other):
		return self.value - other.value

	def __add__(self, other):
		return self.value + other.value

	def __mul__(self, other):
		return self.value * other.value

	def __truediv__(self, other):
		if isinstance(other, variableData):
			return self.value / other.value
		else:
			return self.value / other





@dataclass
class Process_data(Protocol):
	term_critteria: List[Callable] = field(init=False)
	term_type: List[int] = field(init=False)
	term_status: List[bool] = field(init=False) 
	variables: List[variableData]
	responses: List[variableData]

class coordinator(Protocol):

	def clone_point(self):
		...
	
	def create_linking_list(self):
		...

	def calc_inconsistency(self):
		...

	def calc_penalty(self):
		...

	def update_multipliers(self):
		...

class process(Process_data, Protocol):
	
	def run(self):
		...
	
	def validation(self):
		...
	
	def setInputs(self):
		...

	def getOutputs(self):
		...
	
	def setup(self):
		...

class search(Protocol):
	
	def evaluateSamples(self):
		...
	
	def run(self):
		...

@dataclass
class ModelInadequacyData:
	type: int
	relative_inadequacies: np.ndarray
	absolute_inadequacies: np.ndarray
	approx_rel_inadeq: np.ndarray
	approx_abs_inadeq: np.ndarray
	reference_model_index: int
	errorSurrogateType: int


@dataclass
class DA_Data:
	inputs: List[variableData]
	outputs: List[variableData]
	blackbox: Callable
	links: List[int]
	coupling_type: List[int]
	preCondition: Optional[Callable] = field(init=False)
	runningCondition: Optional[Callable]  = field(init=False)
	postCondition: Optional[Callable] = field(init=False)
	model: Optional[object] = field(init=False)
	modelType: Optional[int] = field(init=False)
	validation_list: Optional[List[Callable]]  = field(init=False)
	validation_type: Optional[List[int]] = field(init=False)
	validation_status: Optional[List[bool]] = field(init=False) 

@dataclass
class optimizationData:
	objectives: List[Callable]
	constraints: List[Callable]
	objectiveWeights: List[Any]
	constraintsHandling: int
	solver: Callable

@dataclass
class DA(DA_Data):

	def setup(self, input):
		data: Dict = {}
		if isfile(input):
			data = json.load(input)
		
		if isinstance(input, dict):
			self(**data)
	
	def run(self):
		outs = self.blackbox(self.getInputsList())
		self.setOutputsValue(outs)
		return outs
	
	def validation(self, vType: int):
		self.validation_status = []
		for i in range(len(self.validation_list)):
			if self.validation_type[i] == vType:
					self.validation_status.append(self.validation_list[i])
		
		return all(self.validation_status)
				
	def setInputsValue(self, values: List[Any]):
		for i in range(len(self.inputs)):
			self.inputs[i].value = copy.deepcopy(values[i])

	def setOutputsValue(self, values: Any):
		if len(self.outputs) > 1 and isinstance(values, list):
			for i in range(len(self.outputs)):
				self.outputs[i].value = copy.deepcopy(values[i])
		elif len(self.outputs) == 1 and not isinstance(values, list):
			self.outputs[0].value = copy.deepcopy(values)


	def getInputsList(self):
		v = []
		for i in range(len(self.inputs)):
			v.append(self.inputs[i].value)
		return v

	def getOutputs(self):
		return self.outputs
	
	def getOutputsList(self):
		o = []
		for i in range(len(self.outputs)):
			o.append(self.outputs[i].value)
		return o



@dataclass
class MDA_data(Process_data):
	nAnalyses: int
	analyses: List[DA]

@dataclass
class MDA(MDA_data):

	def setup(self, input):
		data: Dict = {}
		if isfile(input):
			data = json.load(input)
		
		if isinstance(input, dict):
			self(**data)

	def run(self):
		for i in range(self.nAnalyses):
			for j in range(len(self.variables)):
				for k in range(len(self.analyses[i].inputs)): 
					if self.analyses[i].inputs[k].index == self.variables[j].index:
						self.analyses[i].inputs[k] = copy.deepcopy(self.variables[j])
			self.analyses[i].run()
	
	def validation(self, vType: int):
		self.term_status = []
		for i in range(len(self.term_critteria)):
			if self.term_type[i] == vType:
					self.term_status.append(self.term_critteria[i])
	
	def setInputs(self, values: List[Any]):
		self.variables = copy.deepcopy(values)

	def getOutputs(self):
		out = []
		for i in range(len(self.responses)):
			out.append(self.responses[i].value)
		return out

@dataclass
class coordinationData:
	nsp: int
	budget: int = 20
	index_of_master_SP: int = 1
	display: bool = True
	scaling: float = 10.0
	mode: str = "serial"
	var_group: List[variableData] = field(default_factory=list)
	_linker: List[List[int]] = field(default_factory=list)
	n_dim: int = 0	

@dataclass
class ADMM_data(coordinationData):
	beta: float = 1.3
	gamma: float = 0.5
	q: np.ndarray = np.empty([0,0])
	qold: np.ndarray = np.empty([0,0]) 
	phi: float = 1.0
	v: np.ndarray = np.empty([0,0])
	w: np.ndarray = np.empty([0,0])
	update_w: bool = False
	M_update_scheme: int = w_scheme.MEDIAN

# TODO: ADMM needs to be customized for this code
@dataclass
class ADMM(ADMM_data):
	" Alternating directions method of multipliers "
	# Constructor
	def __init__(self, nsp, beta, budget, index_of_master_SP, display, scaling, mode, M_update_scheme):
		""" Initialize the multiplier vectors """
		self.nsp = nsp
		self.beta = beta
		self.budget = budget
		self.index_of_master_SP = index_of_master_SP
		self.display = display
		self.scaling = scaling
		self.mode = mode
		self.started: bool = False
		self.M_update_scheme = M_update_scheme
		


	def clone_point(self, p: variableData):
		self.var_group.append(p)

	def create_linking_list(self):
		a = []
		b = []
		for i in range(self.nsp):
			for j in range(self.nsp):
				if i != j:
					a.append(j)
					b.append(i)

		self._linker.append(a)
		self._linker.append(b)
		

	def calc_inconsistency(self):
		q_temp : np.ndarray = np.empty([0,0])
		for i in range(self._linker.shape[1]):
			if self.master_vars:
				q_temp = np.append(q_temp, subtract(((self.master_vars[self._linker[0, i]-1])),
								((self.master_vars[self._linker[1, i]-1])))*self.scaling)
			else:
				raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
		
		# if self.started:
		# 	self.update_w = not any(np.less_equal(
		# 		np.abs(self.q), self.gamma*np.abs(q_temp)))

		self.q = copy.deepcopy(q_temp)
		
		

	def calc_inconsistency_old(self):
		q_temp : np.ndarray = np.empty([0,0])
		for i in range(self._linker.shape[1]):
			if self.master_vars_old:
				q_temp = np.append(q_temp, subtract(((self.master_vars_old[self._linker[0, i]-1])),
								((self.master_vars_old[self._linker[1, i]-1])))*self.scaling)
			else:
				raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")
		self.qold = copy.deepcopy(q_temp)

	def update_master_vector(self, vars: List[variableData], resps: List[variableData]):
		if self.master_vars:
			for i in range(len(vars)):
				self.master_vars[vars[i].index-1] = copy.deepcopy(vars[i])
			for i in range(len(resps)):
				self.master_vars[resps[i].index-1] = copy.deepcopy(resps[i])
		else:
			raise Exception(IOError, "Master variables vector have to be non-empty to calculate inconsistencies!")


	def calc_penalty(self):
		phi = np.add(np.multiply(self.v, self.q), np.multiply(np.multiply(self.w, self.w), np.multiply(self.q, self.q)))
		self.phi = np.sum(phi)
		
		if np.iscomplex(self.phi) or np.isnan(self.phi):
			self.phi = np.inf

	def update_multipliers(self):
		self.v = np.add(self.v, np.multiply(
						np.multiply(np.multiply(2, self.w), self.w), self.q))
		self.calc_inconsistency_old()
		self.q_stall = np.greater(np.abs(self.q), self.gamma*np.abs(self.qold))
		increase_w = []
		if self.M_update_scheme == w_scheme.MEDIAN:
			for i in range(self.q_stall.shape[0]):
				increase_w.append(2. * ((self.q_stall[i]) and (np.greater_equal(np.abs(self.q[i]), np.median(np.abs(self.q))))))
			# self.update_w = any(np.less_equal(np.abs(self.q), self.gamma*np.abs(self.qold))) 
		elif self.M_update_scheme == w_scheme.MAX:
			increase_w = self.q.shape * (self.q_stall and np.greater_equal(np.abs(self.q), np.max(np.abs(self.q))))
		elif self.M_update_scheme == w_scheme.NORMAL: 
			increase_w = self.q_stall
		elif self.M_update_scheme == w_scheme.RANK: 
			temp = self.q.argsort()
			rank = np.empty_like(temp)
			rank[temp] = np.arange(len(self.q))
			increase_w = np.multiply(np.multiply(2, self.q_stall), np.divide(rank, np.max(rank)))
		else:
			raise Exception(IOError, "Multipliers update scheme is not recognized!")
		
		for i in range(len(self.w)):
			self.w[i] = np.multiply(self.w[i], np.power(self.beta, increase_w[i]))
		
		self.w
	
		

class partitionedProblemData:
	nv: int
	nr: int
	sp_index: int
	vars: List[variableData]
	resps: List[variableData]
	is_main: bool
	MDA_process: process
	coupling: List[float]
	solution: List[Any]
	solver: Any
	realistic_objective: float = False
	optFunctions: List[Callable] = None
	obj: float = np.inf
	constraints: List[float] = [np.inf]
	frealistic: float = 0.
	scaling: float = 10.
	coord: ADMM
	opt: Callable
	fmin_nop: float
	budget: int
	display: bool
	psize: float
	psize_init: int

@dataclass
class SubProblem(partitionedProblemData):
	# Constructor
	def __init__(self, nv, index, vars, resps, is_main, analysis, coordination, opt, fmin_nop, budget, display, psize, pupdate):
		self.nv = nv
		self.index = index
		self.vars = copy.deepcopy(vars)
		self.resps = copy.deepcopy(resps)
		self.is_main = is_main
		self.MDA_process = analysis
		self.coord = coordination
		self.opt = opt
		self.optimizer = OMADS.main
		self.fmin_nop = fmin_nop 
		self.budget=budget
		self.display = display
		self.psize = psize
		self.psize_init = pupdate
	
	def get_minimizer(self):
		v = []
		for i in range(len(self.vars)):
			v.append(self.vars[i])
		return v
	
	def get_coupling_vars(self):
		v = []
		for i in range(len(self.MDA_process.responses)):
			if self.MDA_process.responses[i].coupling_type == COUPLING_TYPE.FEEDFORWARD:
				v.append(self.MDA_process.responses[i])
		return v
	
	def get_design_vars(self):
		v = []
		for i in range(len(self.vars)):
			if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD:
				v.append(self.vars[i])
		return v

	def set_variables_value(self, vlist: List):
		for i in range(len(self.vars)):
			if self.vars[i].coupling_type != COUPLING_TYPE.FEEDFORWARD:
				self.vars[i].value = vlist[i]
	
	def set_pair(self):
		indices1 = []
		indices2 = []
		for i in range(len(self.coord.master_vars)):
			if self.index == self.coord.master_vars[i].sp_index and self.coord.master_vars[i].link and self.coord.master_vars[i].link >= 1:
				linked_to = (self.coord.master_vars[i].link)
				indices1.append(self.coord.master_vars[i].index)
				for j in range(len(self.coord.master_vars)):
					if linked_to == self.coord.master_vars[j].sp_index and self.coord.master_vars[j].name == self.coord.master_vars[i].name:
						indices2.append(self.coord.master_vars[j].index)
		if self.is_main == 1:
			self.coord._linker = copy.deepcopy(np.array([indices1, indices2]))
		else:
			self.coord._linker = copy.deepcopy(np.array([indices2, indices1]))	

	def evaluate(self, vlist: List[float]):
		# If no variables were provided use existing value of the variables of the current subproblem (Might happen during initialization) 
		if vlist is None:
			v: List = self.get_design_vars()
			vlist = self.get_list_vars(v)
		self.set_variables_value(vlist)
		self.MDA_process.setInputs(self.vars)
		self.MDA_process.run()
		y = self.MDA_process.getOutputs()

		fun = self.opt(vlist, y)
		self.fmin_nop = fun[0]
		con = fun[1]
		con = self.get_coupling_vars_diff(con)

		self.set_pair()
		self.coord.update_master_vector(self.vars, self.MDA_process.responses)
		self.coord.calc_inconsistency()
		self.coord.calc_penalty()
		
		if self.realistic_objective:
			con.append(y-self.frealistic)
		return [fun[0]+self.coord.phi, con]
	
	def solve(self, v, w):
		self.coord.v = v
		self.coord.w = w
		# self.coord.master_vars = copy.deepcopy(vars)
		bl = self.get_list_vars(self.get_design_vars())
		eval = {"blackbox": self.evaluate}
		param = {"baseline": bl,
								"lb": self.get_list_vars_lb(self.get_design_vars()),
								"ub": self.get_list_vars_ub(self.get_design_vars()),
								"var_names": self.get_list_vars_names(self.get_design_vars()),
								"scaling": self.get_design_vars_scaling(self.get_design_vars()),
								"post_dir": "./post"}
		# options = {"seed": 0, "budget": self.budget, "tol": 1e-9, "display": self.display, "precision": "high", "psize_init": 1.}
		pinit = min(max(1E-12, self.psize), 1)
		options = {
			"seed": 0,
			"budget": 2*self.budget*len(self.vars),
			"tol": max(pinit/1000, 1E-12),
			"psize_init": pinit,
			"display": self.display,
			"opportunistic": False,
			"check_cache": True,
			"store_cache": False,
			"collect_y": False,
			"rich_direction": False,
			"precision": "high",
			"save_results": False,
			"save_coordinates": False,
			"save_all_best": False,
			"parallel_mode": False
		}
		data = {"evaluator": eval, "param": param, "options":options}

		out = {}
		pinit = self.psize
		# TODO: Coordinator forgets q after calling the optimizer, possible remedy is to update the variables from the optimizer output and the master variables too
		# then at the end of each outer loop iteration we can calculate q of that subproblem before updating penalty parameters
		out = self.optimizer(data)
		if self.psize_init == PSIZE_UPDATE.DEFAULT:
			self.psize = 1.
		elif self.psize_init == PSIZE_UPDATE.SUCCESS: 
			self.psize = out["psuccess"]
		elif self.psize_init == PSIZE_UPDATE.MAX:
			self.psize = out["pmax"] 
		elif self.psize_init == PSIZE_UPDATE.LAST: 
			self.psize = out["psize"]
		else:
			self.psize = 1.
		
		# print(f'sp{self.index} || pinit= {pinit} || psize={self.psize} || bl={bl} || xmin={out["xmin"]}')

		#  We need this extra evaluation step to update inconsistincies and the master_variables vector
		self.evaluate(out["xmin"])
		self.coord.update_master_vector(self.vars, self.MDA_process.responses)
		return out

	def get_coupling_vars_diff(self, con):
		vc: List[variableData] = self.get_coupling_vars()
		# conl = []
		for i in range(len(vc)):
			con.append(float(vc[i].value-vc[i].ub))
		
		for i in range(len(vc)):
			con.append(float(vc[i].lb-vc[i].value))
		return con
	
	def set_dependent_vars(self, vars: List[variableData]):
		for i in range(len(vars)):
			if vars[i].link == self.index:
				if vars[i].coupling_type == COUPLING_TYPE.FEEDFORWARD or vars[i].coupling_type == COUPLING_TYPE.SHARED:
					for j in range(len(self.vars)):
						if self.vars[j].name == vars[i].name:
							self.vars[j].value = vars[i].value

	def get_list_vars(self, vars:List[variableData]):
		v = []
		for i in range(len(vars)):
			v.append(vars[i].value)
		return v
	
	def get_list_vars_ub(self, vars:List[variableData]):
		v = []
		for i in range(len(vars)):
			v.append(vars[i].ub)
		return v

	def get_list_vars_lb(self, vars:List[variableData]):
		v = []
		for i in range(len(vars)):
			v.append(vars[i].lb)
		return v

	def get_design_vars_scaling(self, vars:List[variableData]):
		v = []
		for i in range(len(self.vars)):
			v.append(self.vars[i].scaling)
		return v

	def get_list_vars_names(self,vars:List[variableData]):
		v = []
		for i in range(len(vars)):
			v.append(vars[i].name)
		return v

@dataclass
class MDO_data(Process_data):
	Architecture: int
	Coordinator: ADMM
	subProblems: List[SubProblem]
	fmin: float
	hmin: float
	display: bool
	inc_stop: float
	stop: str
	tab_inc: List
	noprogress_stop: int

	
@dataclass
class MDO(MDO_data):

	def setup(self, input):
		data: Dict = {}
		if isfile(input):
			data = json.load(input)
	
	def get_list_of_var_values(x: List[variableData]):
		x_temp = []
		for i in range(len(x)):
			x_temp.append(x[i].value)
		return x_temp
	
	def get_master_vars_difference(self):
		dx = []
		x =[]
		xold =[]
		for i in range(len(self.Coordinator.master_vars)):
			dx.append(self.Coordinator.master_vars[i] - self.Coordinator.master_vars_old[i])
			x.append(self.Coordinator.master_vars[i].value)
			xold.append(self.Coordinator.master_vars_old[i].value)
		# print(f'x = {x}')
		# print(f'xold = {xold}')
		return np.linalg.norm(dx, 2)
	
	def check_termination_critt(self, iter):
		if iter > 1 and np.abs(self.tab_inc[iter]) < self.inc_stop:
			print(f'Stop: qmax = {np.max(np.abs(self.Coordinator.q))} < {self.inc_stop}')
			self.stop = "Max inconsitency is below stopping threshold"
			return True

		i = self.noprogress_stop

		if iter > i + 2 and np.log(np.min(self.Coordinator.w) > 6.) and np.less_equal(self.tab_inc[iter-i], np.min(self.tab_inc[iter-i+1:iter])):
			print(f'Stop: no progress after {i} iterations.')
			self.stop = "No progress after several iterations"
			return True
		
		return False

	def run(self):
		""" Run the MDO process """
		for iter in range(self.Coordinator.budget):
			if iter > 0:
				self.Coordinator.master_vars_old = copy.deepcopy(self.Coordinator.master_vars)
			else:
				self.Coordinator.master_vars_old = copy.deepcopy(self.variables)
				self.Coordinator.master_vars = copy.deepcopy(self.variables)
				
				
				
			""" ADMM inner loop """
			for s in range(len(self.subProblems)):
				self.subProblems[s].coord = copy.deepcopy(self.Coordinator)
				if iter == 0:
					self.subProblems[s].set_pair()
					self.Coordinator.v = [0.] * len(self.subProblems[s].coord._linker[0])
					self.Coordinator.w = [1.] * len(self.subProblems[s].coord._linker[0])
				
				# self.subProblems[s].set_dependent_vars(self.Coordinator.master_vars)

				out_sp = self.subProblems[s].solve(self.Coordinator.v, self.Coordinator.w)		
				self.Coordinator = copy.deepcopy(self.subProblems[s].coord)
				if self.subProblems[s].index == self.Coordinator.index_of_master_SP:
					self.fmin = self.subProblems[s].fmin_nop
					self.hmin = out_sp["hmin"]
			
			""" Display convergence """
			dx = self.get_master_vars_difference()
			if self.display:
				print(f'{iter} || qmax: {np.max(np.abs(self.Coordinator.q))} || Obj: {self.fmin} || dx: {dx} || max(w): {np.max(self.Coordinator.w)}')
				index = np.argmax(self.Coordinator.q)
				print(f'Highest inconsistency : {self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].name}_'
					f'{self.Coordinator.master_vars[self.Coordinator._linker[0,index]-1].link} to '
						f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].name}_'
					f'{self.Coordinator.master_vars[self.Coordinator._linker[1,index]-1].link}')
			""" Update LM and PM """
			self.Coordinator.update_multipliers()
			
			""" Stopping criterias """
			self.tab_inc.append(np.max(np.abs(self.Coordinator.q)))
			stop: bool = self.check_termination_critt(iter)

			if stop:
				break		
		return self.Coordinator.q
	
	def validation(self, vType: int):
		self.term_status = []
		for i in range(len(self.term_critteria)):
			if self.term_type[i] == vType:
					self.term_status.append(self.term_critteria[i])
	
	def setInputs(self, values: List[Any]):
		self.variables = copy.deepcopy(values)

	def getOutputs(self):
		return self.responses

def termTest():
	print("Termination criterria work!")

def A1(x):
	LAMBDA = 0.0
	if not all(v > 0. for v in x):
		return np.inf
	return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

def A2(x):
	LAMBDA = 0.0
	if not all(v > 0. for v in x):
		return np.inf
	return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

def opt1(x, y):
	return [sum(x)+y[0], [0.]]

def opt2(x, y):
	return [0., [x[1]+y[0]-10.]]

def Basic_MDO():
	#  Variables setup
	# TODO: will be simplified when the N2 chart UI is implemented
	v = {}
	V: List[variableData] = []
	names = ["u", "v", "a", "b", "u", "w", "a", "b"]
	spi = [1,1,1,1,2,2,2,2]
	links = [2, None, 2, 2, 1, None, 1, 1]
	coupling_t = [COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED, COUPLING_TYPE.FEEDFORWARD,
	COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED,
	 COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.FEEDFORWARD]
	lb = [0.]*8
	ub = [10.]*8
	bl = [5.]*8
	scaling = [10.] * 8

	# Variables dictionary with subproblems link
	for i in range(8):
		v[f"var{i+1}"] = {"index": i+1,
		"sp_index": spi[i],
		f"name": names[i],
		"dim": 1,
		"value": 0., 
		"coupling_type": coupling_t[i], 
		"link": links[i],
		"baseline": bl[i],
		"scaling": scaling[i],
		"lb": lb[i],
		"value": bl[i],
		"ub": ub[i]}

	for i in range(8):
		V.append(variableData(**v[f"var{i+1}"]))

	# Analyses setup; construct disciplinary analyses
	DA1: process = DA(inputs=[V[0], V[1], V[3]],
	outputs=[V[2]],
	blackbox=A1,
	links=2,
	coupling_type=COUPLING_TYPE.FEEDFORWARD
	)

	DA2: process = DA(inputs=[V[4], V[5], V[6]],
	outputs=[V[7]],
	blackbox=A2,
	links=1,
	coupling_type=COUPLING_TYPE.FEEDFORWARD
	)

	# MDA setup; construct subproblems MDA
	sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[3]], responses=[V[2]])
	sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6]], responses=[V[7]])

	# Construct the coordinator
	coord = ADMM(beta = 1.3,
	nsp=2,
	budget = 50,
	index_of_master_SP=1,
	display = True,
	scaling = 0.1,
	mode = "serial",
	M_update_scheme= w_scheme.MEDIAN
	)

	# Construct subproblems
	sp1 = SubProblem(nv = 3, 
	index = 1, 
	vars = [V[0], V[1], V[3]],
	resps = [V[2]],
	is_main=1, 
	analysis= sp1_MDA,
	coordination=coord,
	opt=opt1,
	fmin_nop=np.inf,
	budget=20,
	display=False,
	psize = 1.,
	pupdate=PSIZE_UPDATE.LAST)

	sp2 = SubProblem(nv = 3, 
	index = 2, 
	vars = [V[4], V[5], V[6]],
	resps = [V[7]],
	is_main=0, 
	analysis= sp2_MDA,
	coordination=coord,
	opt=opt2,
	fmin_nop=np.inf,
	budget=20,
	display=False,
	psize = 1.,
	pupdate=PSIZE_UPDATE.LAST
	)
	
	# Construct MDO workflow 
	# inc_stop: float
	# stop: str
	# tab_inc: list
	# noprogress_stop: int
	MDAO: MDO = MDO(
	Architecture = MDO_ARCHITECTURE.IDF,
	Coordinator = coord,
	subProblems = [sp1, sp2],
	variables = V,
	responses = [V[2], V[7]],
	fmin = np.inf,
	hmin = np.inf,
	display = True,
	inc_stop = 1E-9,
	stop = "Iteration budget exhausted",
	tab_inc = [],
	noprogress_stop = 100
	)
	
	# Run the MDO problem
	out = MDAO.run()

	print(f'------Run_Summary------')
	print(MDAO.stop)
	print(f'q = {MDAO.Coordinator.q}')
	for i in MDAO.Coordinator.master_vars:
		print(f'{i.name}_{i.sp_index} = {i.value}')
	print(f'Final obj value of the main problem: \n {MDAO.fmin}')

def speedReducer():
	#  Variables setup
	# TODO: will be simplified when the N2 chart UI is implemented
	f1min = 722
	f1max = 5408
	f2min = 184
	f2max = 506
	f3min = 942
	f3max = 1369
	# 
	v = {}
	V: List[variableData] = []
	s  = COUPLING_TYPE.SHARED
	ff = COUPLING_TYPE.FEEDFORWARD
	fb = COUPLING_TYPE.FEEDBACK
	un = COUPLING_TYPE.UNCOUPLED


	names = ["x1", "x1", "x1", "x2", "x2", "x2", "x3", "x3", "x3", "f1", "f2", "x4", "x6", "f3", "x5", "x7", "f1", "f2", "f3"]
	spi =   [ 1,      2,    3,		1,		2,		3,		1,		2,		3,		1,    2,    2,		2,		4,    3,	  3,		4,		4,		4]
	links = [ 2,  [1,3],    2,    2,[1,3],    2, 	  2,[1,3],    2,    4,    4, None, None,    3, None, None, 		1, 		2, 		3]
	coupling_t = \
					[ s,      s,		s,		s,		s,		s,		s,		s,		s,	 ff,   ff,    s,    s,   ff,    s,    s,   fb,   fb,   fb]
	lb = [0.]*19
	ub = [10.]*8
	bl = [5.]*8
	scaling = [10.] * 8

	# Variables dictionary with subproblems link
	for i in range(8):
		v[f"var{i+1}"] = {"index": i+1,
		"sp_index": spi[i],
		f"name": names[i],
		"dim": 1,
		"value": 0., 
		"coupling_type": coupling_t[i], 
		"link": links[i],
		"baseline": bl[i],
		"scaling": scaling[i],
		"lb": lb[i],
		"value": bl[i],
		"ub": ub[i]}

	for i in range(8):
		V.append(variableData(**v[f"var{i+1}"]))


if __name__ == "__main__":
	Basic_MDO()
	







					





