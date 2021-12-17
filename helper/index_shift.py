import numpy as np
import regex as re

def index_shift(string, shift):
    """Shift indices in strings of form "a[1][2]*b[3] + c[4]"

    Args:
        string (str): String of definition
        shift (int): index shift value

    Returns: (str) Input string with indices shifted

    """
    sep = []
    # Separate String into Indces as pure ints and rest as strings
    splitted = string.split("[")
    for st in splitted:
        s = st.split("]")
        for x in s:
            sep.append(x)
    for i in range(len(sep)):
        try:
            sep[i] = "["+str(int(sep[i])+shift)+"]"
        except:
            continue
    out = ""
    for x in sep:
        out = out+x

    return out





string = "3 * a[1] * b[2] + 2*c[1]"
index = -1
index_shift(string, index)

