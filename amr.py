# Required libraries
import numpy as np
import random
import math 
import sys

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

def eprint(*args, **kwargs):
	print(*args,file=sys.stderr,**kwargs)

class AMR(FloatProblem):

	maxAB = 0  # Maximum total antibiotic dose
	samples = 0  # Number of times to run the mathematical model
	interventions = 0  # Number of times able to apply a dose
	maxTime = 192
	tstep = 0.25  # 1/4 hour basic time step
	relativeDose=True
	relativeTime=True

	poissonApprox = True

	def __init__(self, interventions, mxAB, samples, objectives,maxt,lowerBounds,upperBounds):
		
		self.maxAB = mxAB
		self.samples = samples
		self.maxTime = maxt

		# Account for time and dose per intervention
		self.interventions = interventions
		self.number_of_variables = interventions * 2
		self.number_of_objectives = objectives
		self.number_of_constraints = 0

		self.obj_directions = [self.MINIMIZE]
		self.obj_labels = ['f(x)']

		self.lower_bound = lowerBounds
		self.upper_bound = upperBounds 
		# [time of dose 1, size of dose 1, time between dose 1 and dose 2, size of dose 2, ...]
		# self.lower_bound = [2, 0, 0, 0, 0, 0]  # Entry 1 set as 2 hours, rest as zero.
		# self.upper_bound = [2.001, 1, 22, 1, 24, 1]  # Entry 1 keeps first dose at 2 hours
		# self.upper_bound = [24 for _ in range(number_of_variables)]

		FloatSolution.lower_bound = self.lower_bound
		FloatSolution.upper_bound = self.upper_bound
		self.create_solution()

	def printSchedule(self,solution,times,doses):
		#eprint(solution)
		dose=0.0
		eprint("Time\tDose")
		for t in range(0,len(times)):
			eprint("%.2f\t%.2f" % (times[t],doses[t]))
			dose=dose+doses[t]
		#eprint("Total Dose: %.3f" % (dose))
		#eprint("Objectives: {}".format(solution.objectives))

	def poissonApprox(self,lam):
		return  random.gauss(lam,math.sqrt(lam))
	
	def poisson(self, val):
		if self.poissonApprox:
			return self.poissonApprox(val)
		else:
			return np.random.poisson(val)
	
	def getSchedule(self,solution,maxDose):
		doses = int(len(solution.variables)/2)
		
		dose_time = [0] * doses		# [0, 0, 0]  # times of doses (first and last entries fixed at 0 and 192)
		dose_quant = [0] * doses	# [0, 0, 0]  # antibiotic doses (first and last entries fixed at 0)

		availableDose = maxDose
		ctime = 0
		totalDose = 0
		for d in range(0, doses):
			dose_time[d] = ctime + solution.variables[d * 2]
			if self.relativeTime: ctime = dose_time[d]
			if self.relativeDose:
				dose = solution.variables[(d * 2)+1] * availableDose
				availableDose = availableDose - dose
			else:
				dose = solution.variables[(d * 2)+1]
			totalDose += dose
			dose_quant[d] = dose

		return (dose_time,dose_quant)

			
	def evaluate(self, solution: FloatSolution) -> FloatSolution:
		# Model parameters that are mostly fixed
		r_mu = 0.4779  # Average Replication rate of bacteria
		m_mu = 0.6772  # Co-efficient for the host immune response
		n = 0.9193  # Hill co-efficient in the immune response
		v = 0.0525  # Standard deviation for host heterogeneity
		a1 = 0.7281  # Maximum kill rate of antibiotic
		a2 = 0.1910  # Level of antibiotic giving half max kill rate
		k = 2.9821  # Hill co-efficient in AB induced death.
		a = 0.1174  # Decay rate of antibiotic (half-life = 5.9hrs)
		Bdead = 10 ** 9  # Bacterial load at which the host dies [38]
		bacteria_init = 10 ** 5  # Initial bacteria count
		antibiotic_init = 0  # Initial antibiotic concentration
		multi = 200
		a22k = a2 ** k

		# Test treatment vectors for Search Algorithm
		# solution.variables = [2.001,0.5,22.0001,1.0,24.0001,0] 
		# solution.variables = [2,0.5,22,1.0,24,0] 
		(dose_time,dose_quant) = self.getSchedule(solution,self.maxAB)
		

		doses = len(dose_time)
		totalDose = 0
		highDose = 0
		numRealDose = 0
		cureTime = 0
		for d in dose_quant: 
			totalDose = totalDose + d
			if d > highDose : highDose = d
			if d > 0.05 : numRealDose = numRealDose + 1


		treatmentTime = 0
		for t in range(0,len(dose_time)):
			if dose_quant[t] > 0.05:
				treatmentTime = dose_time[t]

		cures = 0
		deaths = 0
		runs = 0

		# Run the model with a given number of samples
		maxABConcentration = 0
		for _ in range(0, self.samples):
			r = random.gauss(r_mu, v)
			m = random.gauss(m_mu, v)
			bacteria = bacteria_init
			antibiotic = antibiotic_init
			time:float = 0.0
			
			t = 0;
			nextDoseTime = dose_time[t] 
			while (time < self.maxTime and bacteria > 0):
				if nextDoseTime <= time and time <= nextDoseTime+self.tstep:
					# eprint("Dose %d at time %.2f  = %.2f" % (t,time,dose_quant[t]))
					antibiotic += dose_quant[t]
					if antibiotic > maxABConcentration:
						maxABConcentration = antibiotic
					t = t + 1
					if t < doses: 
						# Set next available dose
						nextDoseTime=dose_time[t]
					else:
						# If no doses left, ensure next dose is out with max time
						nextDoseTime = self.maxTime + 1
				
				B_rate_inc = r * bacteria
				ab2k = antibiotic ** k
				B_rate_dec = m * (bacteria ** n) + bacteria * a1 * ab2k / (ab2k + a22k)
				bacteria += self.poisson(self.tstep * B_rate_inc) - self.poisson(self.tstep * B_rate_dec)
				A_rate_dec = a * antibiotic
				antibiotic = max(antibiotic - self.poisson(self.tstep * A_rate_dec * multi) / multi, 0)
				if bacteria < 1:
					cures += 1
					cureTime = cureTime + time
					break
				if bacteria > Bdead:
					deaths += 1
					break
				time += self.tstep
				
			runs = runs + 1
			
		# solution.objectives = [0] * len(solution.objectives)
		solution.objectives[0] = float(deaths) / float(runs)
		solution.objectives[1] = totalDose
		
		# solution.objectives[1] = highDose
		solution.objectives[2] = numRealDose
		# solution.objectives[1] = treatmentTime
		# solution.objectives[1] = maxABConcentration
		# solution.objectives[1] = cureTime / cures
		
		# eprint("Deaths:%.3f Total Dose:%.3f" % (solution.objectives[0],solution.objectives[1]))
		return(solution.objectives)
		
	def describe(self):
		eprint("\nModel")
		eprint("maxAB:%.2f" % (self.maxAB))
		eprint("runs:%.2f" % (self.samples))
		eprint("interventions:%.2f" % (self.interventions))
		eprint("max time:%.2f" % (self.maxTime))
		
	#----------------------------------------------------------------------
	# Returns the result
	def get_name(self) -> str:
		return 'AMR'
