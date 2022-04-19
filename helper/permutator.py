from sympy.combinatorics.partitions import Partition
from sympy.combinatorics.permutations import Permutation
import itertools as it

if __name__ == "__main__":
    file = open("../cga_py/permutations.py", "w")
    # generate File header
    file.write("from multivector import *\n\n")

    base2 = [
        "e_12",
        "e_13",
        "e_1i",
        "e_1o",
        "e_23",
        "e_2i",
        "e_2o",
        "e_3i",
        "e_3o",
        "e_io",
    ]
    base3 = [
        "e_123",
        "e_12i",
        "e_12o",
        "e_13i",
        "e_13o",
        "e_1io",
        "e_23i",
        "e_23o",
        "e_2io",
        "e_3io",
    ]
    base4 = ["e_123i", "e_123o", "e_12io", "e_13io", "e_23io"]
    base5 = ["e_123io"]
    perm2 = list(it.permutations([0, 1]))
    perm3 = list(it.permutations([0, 1, 2]))
    perm4 = list(it.permutations([0, 1, 2, 3]))
    perm5 = list(it.permutations([0, 1, 2, 3, 4]))

    def index_permutator(string, permut):
        """TODO: Docstring for index_permutator.
        Returns: TODO

        """
        indices = string.split("_")
        index_num = len(indices[1])
        permutated = indices[0] + "_"
        for i in permut:
            permutated += indices[1][i]
        return permutated

    file.write("# Permutations of order 2 elements\n")
    for x in perm2:
        sig = Permutation(x).signature()
        if int(sig) == 1:
            for base in base2:
                if base == base2[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base + "; ")
        else:
            for base in base2:
                if base == base2[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base + "; ")
        file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 3 elements\n")
    for x in perm3:
        sig = Permutation(x).signature()
        if int(sig) == 1:
            for base in base3:
                if base == base3[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base + "; ")
        else:
            for base in base3:
                if base == base3[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base + "; ")
        file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 4 elements\n")
    for x in perm4:
        sig = Permutation(x).signature()
        if int(sig) == 1:
            for base in base4:
                if base == base4[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base + "; ")
        else:
            for base in base4:
                if base == base4[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base + "; ")
        file.write("\n")
    file.write("\n\n")

    file.write("# Permutations of order 5 elements\n")
    for x in perm5:
        sig = Permutation(x).signature()
        if int(sig) == 1:
            for base in base5:
                if base == base5[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " =  " + base + "; ")
        else:
            for base in base5:
                if base == base5[-1]:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base)
                else:
                    new = index_permutator(base, x)
                    file.write(new + " = -" + base + "; ")
        file.write("\n")
