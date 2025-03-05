from sympy.combinatorics.partitions import Partition
from sympy.combinatorics.permutations import Permutation
import itertools as it
from cga_py import *


if __name__ == "__main__":
    file = open("../cga_py/permutations_pm.py", "w")
    # generate File header
    file.write("from .multivector import *\n\n")
    file.write("from .base_objects import *\n\n")

    base2 = [
        "e_12",
        "e_13",
        "e_1p",
        "e_1m",
        "e_23",
        "e_2p",
        "e_2m",
        "e_3p",
        "e_3m",
        "e_pm",
    ]
    base3 = [
        "e_123",
        "e_12p",
        "e_12m",
        "e_13p",
        "e_13m",
        "e_1pm",
        "e_23p",
        "e_23m",
        "e_2pm",
        "e_3pm",
    ]
    base4 = ["e_123p", "e_123m", "e_12pm", "e_13pm", "e_23pm"]
    base5 = ["e_123pm"]
    perm2 = list(it.permutations([0, 1]))
    perm3 = list(it.permutations([0, 1, 2]))
    perm4 = list(it.permutations([0, 1, 2, 3]))
    perm5 = list(it.permutations([0, 1, 2, 3, 4]))

    def index_permutator(string, permut):
        """TODO: Docstring for index_permutator.
        Returns: TODO

        """
        indices = string.split("_")
        permutated = indices[0] + "_"
        perm_prod = "1 "
        for i in permut:
            permutated += indices[1][i]
            perm_prod += " * " + indices[0] + "_" + indices[1][i]
        return permutated, perm_prod

    file.write("# Permutations of order 2 elements\n")
    for x in perm2:
        sig = Permutation(x).signature()
        for base in base2:
            new, prod = index_permutator(base, x)
            file.write(new + " = " + repr(eval(prod)))
            file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 3 elements\n")
    for x in perm3:
        sig = Permutation(x).signature()
        for base in base3:
            new, prod = index_permutator(base, x)
            file.write(new + " = " + repr(eval(prod)))
            file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 4 elements\n")
    for x in perm4:
        sig = Permutation(x).signature()
        for base in base4:
            new, prod = index_permutator(base, x)
            file.write(new + " = " + repr(eval(prod)))
            file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 5 elements\n")
    for x in perm5:
        sig = Permutation(x).signature()
        for base in base5:
            new, prod = index_permutator(base, x)
            file.write(new + " = " + repr(eval(prod)))
            file.write("\n")
    file.write("\n\n")
