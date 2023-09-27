def are_floats_equal(float1, float2, epsilon=1e-9):
    return abs(float1 - float2) < epsilon


def is_close_to_zero(float_num, epsilon=1e-9):
    return abs(float_num) < epsilon