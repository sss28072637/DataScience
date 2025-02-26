import numpy as np
import os
import nevergrad as ng
from HomeworkFramework import Function
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)

class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        def objective_function(solution):
            value = self.f.evaluate(func_num, solution)
            if value == "ReachFunctionLimit":
                return float("inf")
            return float(value)

        instrumentation = ng.p.Array(shape=(self.dim,)).set_bounds(self.lower, self.upper)
        optimizer = ng.optimizers.NGOpt(parametrization=instrumentation, budget=FES)

        recommendation = optimizer.minimize(objective_function)
        self.optimal_solution = recommendation.value
        self.optimal_value = objective_function(recommendation.value)


if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500

        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        # change the name of this file to your student_ID and it will output properly
        with open("{}_function{}.txt".format(os.path.basename(__file__).split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 