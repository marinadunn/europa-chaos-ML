# from test_objs.test_evaluator import TestEvaluator
from test_objs.test_cumulative_eval import TestCumulativeEvaluator

if __name__ == "__main__":
    # test_obj = TestEvaluator()
    # test_obj.temp_save()
    # test_obj.run_tests()

    test_obj_cumulative = TestCumulativeEvaluator()
    test_obj_cumulative.temp_save()
    test_obj_cumulative.run_tests()
