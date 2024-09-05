from collections import deque
import numpy as np
import time

class STLEvaluator:
    def __init__(self,signals,formula):
        self.signals=signals
        self.formula=formula
        self.prev_results={}

    def robustness(self,signal_index,t):
        return self.signals[signal_index][t]

    def F_robustness(self,function,signal_index,t):
        robustnesses=[function(tau) for tau in range(t,len(self.signals[signal_index]))]
        return np.max(robustnesses)

    def G_robustness(self,function,signal_index,t):
        robustnesses=[function(tau) for tau in range(t,len(self.signals[signal_index]))]
        return np.min(robustnesses)

    def and_robustness(self,function_1,function_2,t):
        return np.min([function_1(t),function_2(t)])    

    def incremental_robustness(self,function,signal_index,t,prev=None,is_F=True):
        """
        A generic incremental robustness function for both F (Eventually) and G (Globally).
        
        :param function: A lambda function that computes robustness at a given time step.
        :param signal: The signal over which robustness is computed.
        :param t: The current time step.
        :param prev: Previous robustness value and window (tuple).
        :param is_F: If True, computes eventually robustness (F). If False, computes globally robustness (G).
        :return: The updated robustness value and the sliding window.
        """
        if prev is None:
            #if no previous, compute from scratch
            return self.F_robustness(function,signal_index,t) if is_F else self.G_robustness(function,signal_index,t)
        #compute robustness at current timestep
        robustness_value=function(len(self.signals[signal_index])-1)
        if is_F:
            new_val=max(prev,robustness_value)
        else:
            new_val=min(prev,robustness_value)
        
        return new_val
    def evaluate(self,inner_formula,signal_index,t,key,is_F):
        """
        Generic helper function to evaluate F (Eventually) and G (Globally) robustness.
        
        :param inner_formula: The formula being applied.
        :param signal: The signal over which the robustness is computed.
        :param t: The current time step.
        :param prev_results: The dictionary storing previous robustness values and windows.
        :param is_F: If True, computes Eventually robustness (F). If False, computes Globally robustness (G).
        :return: The robustness value at time t.
        """
        result=self.incremental_robustness(inner_formula,signal_index,t,self.prev_results[key].get(t),is_F)
        self.prev_results[key][t]=result
        return result

    def apply_formula(self,formula=None):
        """
        Apply a nested STL formula recursively.

        :param formula: A nested list representing the formula. The list can contain:
                        - "F" for the eventually operator
                        - "and" for the conjunction
                        - signals as integers (indices into the 'signals' list)
        :param signals: A list of signals (trajectories) to be used in the formula.
        :param prev_results: A dictionary storing previous robustness values and windows.
        :return: A lambda function that evaluates the formula at a given time t.
        """
        if formula is None:
            formula=self.formula
        if isinstance(formula,int): #base case:the formula is the index of the signal
            return lambda t: self.robustness(formula,t) 
        
        operator=formula[0]

        if operator in ['F','G']:
            is_F= operator=='F'
            key=(operator,str(formula[1]))
            if key not in self.prev_results:
                self.prev_results[key] = {}
            inner_formula=self.apply_formula(formula[1])
            signal_index=self.get_signal_index(formula[1])
            return lambda t: self.evaluate(inner_formula,signal_index,t,key,is_F=is_F)
        
        elif operator=='and':
            key_left = ('and_left', str(formula[1]))
            key_right = ('and_right', str(formula[2]))
            if key_left not in self.prev_results:
                self.prev_results[key_left] = {}
            if key_right not in self.prev_results:
                self.prev_results[key_right] = {}
            left_formula=self.apply_formula(formula[1])
            right_formula=self.apply_formula(formula[2])
            return lambda t: self.and_robustness(left_formula,right_formula,t)

    def get_signal_index(self, formula):
        """
        Recursively extract the signal index from the formula.
        """
        if isinstance(formula, int):  # base case: it's a signal index
            return formula
        else:
            # If the formula is an operator, we must get the signal index of the subformula
            return self.get_signal_index(formula[1])

    def append_signal(self, signal_index, new_value):
        """
        Extend a specific signal with new data.
        
        :param signal_index: The index of the signal to extend.
        :param new_value: new value to append to the signal.
        """
        self.signals[signal_index].append(new_value)

"""
# Example usage
x = [-1, 0, 1, 2, 3, 4, 5]
y = [-2, -1, 0, 1, 2, 3, 4]
z = [-3, -2, -1, 0, 1, 2, 3]
signals = [x, y, z]

formula = ["F", ["and", 2, ["F", ["and", 1, ["F", 0]]]]]

# Initialize the STL Evaluator
evaluator = STLEvaluator(signals, formula)

# Apply the formula to get the nested function
nested_formula = evaluator.apply_formula()


# First evaluation with the original signals
start=time.time()
print(nested_formula(0)) 
print(f"time: {time.time()-start}")
# Extend the signals
evaluator.append_signal(0, 6)
evaluator.append_signal(1, 5)
evaluator.append_signal(2, 4)

# Re-evaluate with the extended signals
start=time.time()
print(nested_formula(0))
print(f"time: {time.time()-start}")

evaluator.append_signal(0, -1)
evaluator.append_signal(1, -1)
evaluator.append_signal(2, -1)

# Re-evaluate with the extended signals
start=time.time()
print(nested_formula(0))
print(f"time: {time.time()-start}")

signals=[np.array([3, 3, -3, -1, -1])+20]

formula=['G',0]
evaluator = STLEvaluator(signals, formula)

nested_formula=evaluator.apply_formula()
print(nested_formula(0))
"""
