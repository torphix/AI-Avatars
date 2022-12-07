from .headpose_tests import test_headpose


def test(test_type, *args):
    if test_type == 'headpose':
        test_headpose(*args)

