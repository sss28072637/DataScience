import numpy as np
import cma
import os
from HomeworkFramework import Function

class CMAES_optimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def objective_function(self, x):
        value = self.f.evaluate(self.target_func, x)
        if value == "ReachFunctionLimit":
            return float("inf")
        return float(value)
    
    def run(self, FES):
        initial_solution = np.random.uniform(self.lower, self.upper, self.dim)
        es = cma.CMAEvolutionStrategy(initial_solution, 0.5, {'bounds': [self.lower, self.upper], 'maxfevals': FES})
        
        while not es.stop():
            solutions = es.ask()
            values = [self.objective_function(s) for s in solutions]
            es.tell(solutions, values)
            es.disp()
            
            if min(values) < self.optimal_value:
                self.optimal_value = min(values)
                self.optimal_solution = solutions[np.argmin(values)]
        
        es.result_pretty()

if __name__ == '__main__':
    func_num = 1
    fes = 0
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        op = CMAES_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        with open("{}_function{}.txt".format(os.path.basename(__file__).split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
