import numpy as np
import os
import nlopt
from HomeworkFramework import Function

np.random.seed(42)

class RS_optimizer(Function):  # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def objective_function(self, x, grad):
        if grad.size > 0:
            grad[:] = 0.0
        value = self.f.evaluate(self.target_func, x)
        self.eval_times += 1
        # print(f"Evaluating at {x} -> {value}")

        if value == "ReachFunctionLimit":
            print("ReachFunctionLimit encountered.")
            return float("inf")
        
        return float(value)

    def run(self, FES):  # main part for your implementation
        opt = nlopt.opt(nlopt.GN_DIRECT_L, self.dim)
        # opt = nlopt.opt(nlopt.GN_CRS2_LM, self.dim)
        lower_bounds = np.full(self.dim, self.lower)
        upper_bounds = np.full(self.dim, self.upper)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        opt.set_min_objective(self.objective_function)
        opt.set_maxeval(FES)

        initial_solution = np.random.uniform(lower_bounds, upper_bounds, self.dim)
        # print(f"Initial solution: {initial_solution}")
        # print(f"Lower bounds: {lower_bounds}")
        # print(f"Upper bounds: {upper_bounds}")

        try:
            self.optimal_solution = opt.optimize(initial_solution)
            self.optimal_value = opt.last_optimum_value()
        except Exception as e:
            print(f"Optimization terminated with exception: {e}")

        print("optimal: {}\n".format(self.optimal_value))

if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
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
