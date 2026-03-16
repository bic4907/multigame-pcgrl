from glob import glob
from os.path import abspath, join, dirname

from pcgrllm.evaluation.solution import SolutionEvaluator
from pcgrllm.task import TaskType
from pcgrllm.utils.storage import Iteration

iteration_list = glob(join(dirname(__file__), 'inputs', '*', 'iteration_*'))

# filter that string includes "pe-cot_it-6_fit-hr_exp-def_t-sce_chr-1_1_s-3"
iteration_list = [path for path in iteration_list if "pe-cot_it-6_fit-hr_exp-def_t-sce_chr-1_1_s-1" in path]
# sort the iteration list
iteration_list.sort()

for iteration_path in iteration_list:
    iteration = Iteration.from_path(iteration_path)

    evaluator = SolutionEvaluator(task=TaskType.Scenario, )
    result = evaluator.run(iteration=iteration, scenario_num=1)

    print(iteration_path)
    print(result.exist_imp_perc, result.acc_imp_perc, result.reach_imp_perc)
    # EvaluationResult(playability=0.800000011920929, path_length=26.25, solvability=0.699999988079071, n_solutions=3.299999952316284, loss_solutions=2.700000047683716, reach_imp_perc=0.0, exist_imp_perc=0.0, acc_imp_perc=2.0, fp_imp_perc=0.0, fn_imp_perc=1.0, tp_imp_perc=0.0, tn_imp_perc=0.0, task=scenario, sample_size=10)

    assert 0 <= result.playability <= 1
    assert 0 <= result.path_length <= 100
    assert 0 <= result.solvability <=1
    assert 0 <= result.n_solutions <= 30
    assert 0 <= result.loss_solutions <= 30
    assert 0 <= result.reach_imp_perc <= 1
    assert 0 <= result.exist_imp_perc <= 1
    assert 0 <= result.acc_imp_perc <= 3
    assert 0 <= result.fp_imp_perc <= 1
    assert 0 <= result.fn_imp_perc <= 1
    assert 0 <= result.tp_imp_perc <= 1
    assert 0 <= result.tn_imp_perc <= 1

    print()