import sys
import io
import timeit
import time

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem.multiobjective.unconstrained import OneZeroMax
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.core.solution import BinarySolution, FloatSolution


from jmetal.util.termination_criterion import StoppingByEvaluations
from amr import AMR
from examples.singleobjective import genetic_algorithm
from jmetal import algorithm
from pickle import TRUE

def eprint(*args, **kwargs):
	print(*args,file=sys.stderr,**kwargs)
	
class Solver:
	
	prm = {
		'task':'test',
		
		'samples':0,
		'maxAB':0,
		'maxTime': 0,
		'variables':0,
		'interventions':0,
		
		'popSize':0,	
		'generations':0,
		
		'testRuns':1,
		'trace':0,
		
		'lowerBounds':[2.0,0.0],
		'upperBounds':[2.01,1]
		}
	
	prm['maxEval'] = prm['generations']*prm['popSize']
	objectives = 0

	def makeBoundsArray(self,bounds):
		variables = bounds.split(',')
		vbounds = [0] * len(variables)
		for i in range(0,len(variables)):
			vbounds[i] = float(variables[i])
		return vbounds
	
	def assignParams(self,lines):
		for fl in lines:
			l = fl.strip()
			if len(l) == 0:	continue
			if (l.startswith("//")): continue
			toks=l.split("=")
			if (len(toks) < 2): continue
			
			# print("%s = %s" % (toks[0],toks[1]))
			var = toks[0].strip()
			if (var == 'lowerBounds' or var == 'upperBounds'):
				self.prm[var] = self.makeBoundsArray(toks[1].strip())
				continue
			
			try:
				self.prm[var] = float(toks[1].strip())
			except Exception as e:
				self.prm[var] = toks[1].strip()
				
				
		self.prm['variables'] = self.prm['interventions']*2		
		self.prm['maxEval'] = self.prm['generations']*self.prm['popSize']
				
	def tracing(self):
		return self.prm['trace']==1
	
	def loadParams(self,paramFile):
		f = io.open(paramFile)
		lines = f.readlines()
		f.close()
		self.assignParams(lines)

	def printParams(self):
		for p in self.prm:	
			print("{}:{}".format(p,self.prm[p]))
	
	def printResults(self,problem,algorithm,result,runtime):
		eprint("\nAlgorithm")
		eprint("Population: %.0f" % (self.prm['popSize']))
		eprint('Algorithm: {}'.format(algorithm.get_name()))
		eprint('Problem: {}'.format(problem.get_name()))
		#eprint('Fitness: {}'.format(result.objectives))
		#eprint('Solution {}'.format(result.variables))
		eprint('Computing time: {}'.format(algorithm.total_computing_time))
		eprint("Mean Cost = %.3f sec/ eval" % (algorithm.total_computing_time/self.prm['maxEval']))
		eprint("Max loops per model evaluation %.0f" % ((problem.maxTime/problem.tstep)*self.prm['maxEval']))
		
		if type(result) is not list:
			result = [result]
		
		for r in result:
			(doseT,doseQ) = problem.getSchedule(r,self.prm['maxAB'])
			problem.printSchedule(r,doseT,doseQ)

	def logResults(self,problem,objectives,result,runtime):
		
		#sols = algorithm.solutions;
		#for s in sols:
		#	print(s)

		# Scores
		first = TRUE
		#print('[',end="", flush=True)
		for o in objectives:
			if not first: print(",",end="",flush=True)
			first = False
			print("%.4f" % (o), end="")
		print(",\t\t",end="")
		# print('{}:'.format(result.objectives),end="")
		
		# Solution
		(doseT,doseQ) = problem.getSchedule(result,self.prm['maxAB'])
		doses = len(doseT)
		# print('[',end="", flush=True)
		first = True
		for d in range(0,doses):
			if not first: print(', ',end="", flush=True)
			print("%.2f,%.4f" % (doseT[d],doseQ[d]),end="", flush=True)
			first = False
		print("\t,%.2f,%d" % (runtime,int(ev.prm['samples'])),flush=True)
		#print()
		#print("]", flush=True)
		
			
	def solveGA(self,problem):
		tic = timeit.default_timer()
		algorithm = GeneticAlgorithm(
			problem=problem,
			population_size=int(self.prm['popSize']),
			offspring_population_size=int(self.prm['popSize']),
			mutation=PolynomialMutation(1.0 / problem.number_of_variables, distribution_index=20.0),
			crossover=SBXCrossover(1.0, distribution_index=20.0),
			selection=BinaryTournamentSelection(),
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval']))
		)
		algorithm.run()
		if self.tracing():
			self.printResults(problem,algorithm)
			eprint("",flush=True)
			time.sleep(0.5)
		toc = timeit.default_timer()
		results = algorithm.get_result()
		self.logResults(problem,results.objectives,results,toc - tic)
			
				
	def solveNSGA2(self,problem):
		tic = timeit.default_timer()
		# binary_string_length = 32
		#problem = OneZeroMax(binary_string_length)

		algorithm = NSGAII(
			problem=problem,
			population_size=int(self.prm['popSize']),
			offspring_population_size=int(self.prm['popSize']),
			mutation=PolynomialMutation(1.0 / problem.number_of_variables, distribution_index=20.0),
			crossover=SBXCrossover(1.0, distribution_index=20.0),
			# crossover=SPXCrossover(probability=1.0),
			termination_criterion=StoppingByEvaluations(max_evaluations=int(self.prm['maxEval']))
			)

		algorithm.run()
		front = algorithm.get_result()
		# Save results to file
		#print_function_values_to_file(front, 'FUN.' + algorithm.label + ".txt")
		#print_variables_to_file(front, 'VAR.'+ algorithm.label + ".txt")
		
		toc = timeit.default_timer()
		if type(front) is not list:
			front = [front]
		for p in front:
			self.logResults(problem,p.objectives,p,toc - tic)
		print()

	def testAMR(self,problem):
		solution = FloatSolution(problem.lower_bound,problem.upper_bound,problem.number_of_objectives)

		
		# dose_time = [0, 2, 24, 48, 192]
		# dose_quant = [0, 0.9, 0, 0, 0]   # antibiotic doses (first and last entries fixed at 0)
		# dose_quant = [0, 0.45, 0.45, 0, 0]   # antibiotic doses (first and last entries fixed at 0)
		# dose_quant = [0, 0.3, 0.3, 0.3, 0]   # antibiotic doses (first and last entries fixed at 0)

		# Are the timings and doses relative to previous time and available dose or not
		relative = True		
		problem.relativeTime = relative
		problem.relativeDose = relative

		if relative:
			# Equivalent relative timings and doses for previous schedules			
			#solution.variables = [2,1.0,22,0.0,24,0] 		# Equivalent to [2,0.9,  24,0,	48,0] with 0.9 max dose
			solution.variables = [2.01,0.5,22.01,1.0,24.1,0] 		# Equivalent to [2,0.45, 24,0.45, 48,0] with 0.9 max dose
			# solution.variables = [2,0.3333,22,0.5,24,1.0] 	# Equivalent to [2,0.3,  24,0.3,  48,0.3] with 0.9 max dose
		else:
			# Treatment vectors for previous traditional treatments
			#solution.variables = [2,0.9,24,0.0,48,0]
			solution.variables = [2,0.45,24,0.45,48,0]
			#solution.variables = [2,0.3,24,0.3,48,0.3]
			#solution.variables = [2.00,0.8599, 13.65,0.0307, 27.08,0.0037]
			
		(doseT,doseQ) = problem.getSchedule(solution,self.prm['maxAB'])
		problem.printSchedule(solution,doseT,doseQ)
		# eprint(solution.variables)
		mean = 0;
		tic = timeit.default_timer()
		runs = int(self.prm['testRuns'])
		for e in range(0,runs):
			res = problem.evaluate(solution)
			mean = mean + res[0]
		meanDeaths = mean/runs
		
		toc = timeit.default_timer()
		run_time = (toc - tic)   # time in seconds#

		eprint("Relative Schedule: {}".format(relative))
		eprint("Max Time: {}".format(self.prm['maxTime']))
		eprint("Mean Deaths:%.3f\nMean Survive:%.3f" % (meanDeaths,1-meanDeaths))
		eprint("Test runs: %d with %d samples per evaluation" % (runs,problem.samples))
		eprint("Seconds / Evaluation :%.4f" % (run_time/runs));
		eprint("Loops per model evaluation %.0f" % ((problem.maxTime/problem.tstep)*self.prm['maxEval']))	

	def checkCommands(self,args):
		if len(args) < 1: return
		self.assignParams(args)
		
	def processTask(self,options):
		# The first command line parameter must be a parameter file specifying the task
		if len(options)<=1:
			print("Parameter file required:")
			print("  solver amr-params.txt x=y y=z")
			return
		
		# Load the task parameters from the supplied param file
		self.loadParams(options[1])
		# Now override file parameters with possible command line parameters 
		self.checkCommands(options)

		# Set up our AMR model and describe it
		problem = AMR(
			int(ev.prm['interventions']),
			ev.prm['maxAB'],
			int(ev.prm['samples']),
			int(ev.prm['objectives']),
			int(ev.prm['maxTime']),
			ev.prm['lowerBounds'],
			ev.prm['upperBounds']
			)
				
		# What's our task?
		task = ev.prm['task']
		if (self.tracing()): # if trace is true, print out the config for this task 
			eprint("Task: %s" % (task))
			problem.describe()
			# self.printParams()

		# Now carry out the task		
		if (task == "test"): self.testAMR(problem)
		if (task == "GA"): self.solveGA(problem)
		if (task == "NSGA2"): self.solveNSGA2(problem)		



ev = Solver()
ev.processTask(sys.argv)
