from tests.test_cumulative_eval import TestCumulativeEvaluator

if __name__ == "__main__":

    test_obj_cumulative = TestCumulativeEvaluator()  # create the test object
    test_obj_cumulative.temp_save()  # save the results to a temp file
    test_obj_cumulative.run_tests()  # run the tests
