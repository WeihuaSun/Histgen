def are_floats_equal(float1, float2, epsilon=1e-6):
    return abs(float1 - float2) < epsilon


def is_close_to_zero(num, epsilon=1e-6):
    if isinstance(num, int):
        return num == 0
    else:
        return abs(num) < epsilon
