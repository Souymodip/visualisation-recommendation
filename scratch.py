

def test_network():
    from test_cases import network_decisions_test
    network_decisions_test.main()

def test_timeseries():
    from test_cases import timeseries_decision_test
    timeseries_decision_test.main()

def test_venn():
    from test_cases import venn_check_tests
    venn_check_tests.main()


def test_hierarchy():
    from test_cases import hierarchy_decision_tests
    hierarchy_decision_tests.main()

def main():
    # test_network()
    # test_timeseries()
    # test_venn()
    test_hierarchy()


if __name__ == '__main__':
    main()