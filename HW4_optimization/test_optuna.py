import numpy as np
import os
import optuna
from HomeworkFramework import Function

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

    def objective_function(self, trial):
        solution = np.array([trial.suggest_uniform(f'x{i}', self.lower, self.upper) for i in range(self.dim)])
        value = self.f.evaluate(self.target_func, solution)
        self.eval_times += 1

        if value == "ReachFunctionLimit":
            trial.study.stop()  # Stop the study if the function limit is reached
            return float("inf")

        if float(value) < self.optimal_value:
            self.optimal_solution[:] = solution
            self.optimal_value = float(value)
        
        return float(value)

    def run(self, FES): # main part for your implementation
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective_function, n_trials=FES)
        self.optimal_solution = np.array([study.best_params[f'x{i}'] for i in range(self.dim)])
        self.optimal_value = study.best_value

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
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(os.path.basename(__file__).split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 
