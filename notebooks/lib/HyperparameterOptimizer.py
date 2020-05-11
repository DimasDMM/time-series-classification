import copy

class HyperparameterOptimizer:
    # This is a recursive function, ignore the last parameter
    def tunning_params_combinations(self, tunning_params, combination={}):
        if len(tunning_params) == 0:
            return [combination]

        c_combinations = []

        # Get first param
        param_name = next(iter(tunning_params))
        param_values = tunning_params[param_name]

        for v in param_values:
            c = copy.deepcopy(combination)
            sub_tp = copy.deepcopy(tunning_params)

            del sub_tp[param_name]
            c[param_name] = v

            c_combinations = c_combinations + self.tunning_params_combinations(sub_tp, c)
        
        return c_combinations
