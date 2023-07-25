#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Nikolett Biró'
__email__ = 'bironiki97@gmail.com ; biro.nikolett@wigner.hu'

from math import factorial
from math import pi

def cos_approx(x, accuracy=10):
    """ This function approximates the cosine-function using Taylor expansion.
    Accuracy (int) tells the number of terms ('n') of the series expansion.
    x (float) is the number of which we calculate the cosine of.
    Returns the cosine of x (float).
    """
    assert int(accuracy) == accuracy, "accuracy should be an integer!"
    list_for_series = [(-1)**n/factorial(2*n)*(x**(2*n)) for n in range(accuracy)]
    return (sum(list_for_series))


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    assert cos_approx(0) < 1+1.e-2 and cos_approx(0) > 1-1.e-2, "cos(0) is not 1"

    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
