#Import needed libraryies
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Metropolis():
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.currentState = initialState
        self.samples = list()
        self.step_size = 1

    def __accept(self, proposedState):
        rand = random.random()
        acceptance_probability = min(1, math.exp(self.logTarget(proposedState)/self.logTarget(self.currentState)))
        if(rand < acceptance_probability):
            return(True)
        return(False)

    def adapt(self, blockLengths):
        for total_states in blockLengths:
            num_accept = 0
            state = self.currentState
            for state_inst in range(total_states):
                newState = np.random.normal(state, self.step_size)
                if(self.__accept(newState)):
                    state = newState
                    num_accept += 1
            acceptance_rate = num_accept/total_states
            target_acceptance_rate = 0.4

            self.step_size *= (target_acceptance_rate/acceptance_rate)**1.1
        
        return(self)

    def sample(self, nSamples):
        for i in range(nSamples):
            newState = np.random.normal(self.currentState, self.step_size)
            if(self.__accept(newState)):
                self.currentState = newState
                self.samples.append(newState)
        return(self)
    
    def summary(self):
        summary_dict = dict()
        summary_dict["mean"] = np.mean(self.samples)
        summary_dict["c025"] = np.percentile(self.samples, 2.5)
        summary_dict["c975"] = np.percentile(self.samples, 97.5)
        return(summary_dict)