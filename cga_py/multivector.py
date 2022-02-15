import numpy as np

class cga_object:

    """Element of the CGA with methods for:
    addition, multiplication,
    printing

    Parameters
    ----------

    Returns
    -------


    """
    dim = 32

    coeff_names = ["",
                   "e_1",
                   "e_2",
                   "e_3",
                   "e_i",
                   "e_o",
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
                   "e_123i",
                   "e_123o",
                   "e_12io",
                   "e_13io",
                   "e_23io",
                   "e_123io"]

    even_indices = np.array([ 0,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 26, 27,
                             28, 29, 30])

    def __init__(self, gen=[0], even = False):
        """

        Parameters
        ----------
        gen :
             (Default value = [0])
        even :
             (Default value = False)

        Returns
        -------

        """
        if isinstance(gen,cga_object):
            cof = gen.coeff
        else:
            cof = gen
        ## Version if list is given
        self.coeff = np.zeros(self.dim)
        if not even:
            for i in range(len(cof)):
                self.coeff[i] = cof[i]
        else:
            for i in range(len(self.even_indices)):
                self.coeff[self.even_indices[i]] = cof[i]

    def __add__(self, other):
        """Addition of cga_object

        Parameters
        ----------
        other : cga_object
            object to add to self
            Returns: addition of cga_object with cga_object

        Returns
        -------

        """
        # This is a rather hacky way of ensuring that addition of different
        # datatypes is commutative. Pending better implementation
        try:
            coefficients = self.coeff + other.coeff
        except:
            coefficients = (cga_object([other])+self).coeff
        return cga_object(coefficients)

    __radd__ = __add__

    def __sub__(self, other):
        """TODO: Docstring for __sub__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return self + (-other)

    __rsub__ = __sub__

    def __mul__(self, other):
        """CGA multiplication of two cga_object

        Parameters
        ----------
        other : cga_object
            cga_object to multiply with self
            Returns: (cga_object) multiplication of other and self

        Returns
        -------

        """
        try:
            out = [
    -2*self.coeff[27]*other.coeff[26] - self.coeff[10]*other.coeff[10] -
            self.coeff[7]*other.coeff[7] + self.coeff[1]*other.coeff[1] +
            self.coeff[2]*other.coeff[2] - self.coeff[6]*other.coeff[6] +
            self.coeff[3]*other.coeff[3] - self.coeff[16]*other.coeff[16] +
            2*self.coeff[12]*other.coeff[11] +
            2*self.coeff[23]*other.coeff[22] -
            2*self.coeff[5]*other.coeff[4] +
            2*self.coeff[20]*other.coeff[19] +
            2*self.coeff[18]*other.coeff[17] +
            2*self.coeff[9]*other.coeff[8] +
            2*self.coeff[14]*other.coeff[13] +
            self.coeff[0]*other.coeff[0],
    2*self.coeff[14]*other.coeff[19] - self.coeff[3]*other.coeff[7] +
            other.coeff[1]*self.coeff[0] - self.coeff[2]*other.coeff[6] +
            2*self.coeff[5]*other.coeff[8] + self.coeff[6]*other.coeff[2] +
            2*self.coeff[27]*other.coeff[22] +
            2*self.coeff[20]*other.coeff[13] -
            2*self.coeff[23]*other.coeff[26] +
            2*self.coeff[12]*other.coeff[17] -
            self.coeff[16]*other.coeff[10] + self.coeff[1]*other.coeff[0] +
            2*self.coeff[18]*other.coeff[11] -
            self.coeff[10]*other.coeff[16] - 2*self.coeff[9]*other.coeff[4]
            + self.coeff[7]*other.coeff[3],
    2*self.coeff[14]*other.coeff[22] - self.coeff[3]*other.coeff[10] +
            self.coeff[2]*other.coeff[0] + other.coeff[2]*self.coeff[0] +
            2*self.coeff[5]*other.coeff[11] - self.coeff[6]*other.coeff[1]
            - 2*self.coeff[27]*other.coeff[19] +
            2*self.coeff[20]*other.coeff[26] +
            2*self.coeff[23]*other.coeff[13] -
            2*self.coeff[12]*other.coeff[4] + self.coeff[1]*other.coeff[6]
            + self.coeff[16]*other.coeff[7] -
            2*self.coeff[18]*other.coeff[8] + self.coeff[10]*other.coeff[3]
            + self.coeff[7]*other.coeff[16] -
            2*self.coeff[9]*other.coeff[17],
    -2*self.coeff[14]*other.coeff[4] + self.coeff[3]*other.coeff[0] +
            2*self.coeff[5]*other.coeff[13] + other.coeff[3]*self.coeff[0]
            + self.coeff[2]*other.coeff[10] +
            2*self.coeff[27]*other.coeff[17] -
            2*self.coeff[20]*other.coeff[8] -
            2*self.coeff[23]*other.coeff[11] -
            2*self.coeff[12]*other.coeff[22] + self.coeff[1]*other.coeff[7]
            - self.coeff[16]*other.coeff[6] -
            2*self.coeff[18]*other.coeff[26] -
            self.coeff[10]*other.coeff[2] - self.coeff[7]*other.coeff[1] -
            2*self.coeff[9]*other.coeff[19] -
            self.coeff[6]*other.coeff[16],
    -self.coeff[17]*other.coeff[6] + 2*self.coeff[30]*other.coeff[22] -
            self.coeff[8]*other.coeff[1] - self.coeff[7]*other.coeff[19] +
            other.coeff[4]*self.coeff[0] + self.coeff[2]*other.coeff[11] -
            2*self.coeff[15]*other.coeff[4] - self.coeff[6]*other.coeff[17]
            + 2*self.coeff[29]*other.coeff[19] -
            self.coeff[22]*other.coeff[10] + self.coeff[26]*other.coeff[16]
            - 2*self.coeff[25]*other.coeff[13] -
            self.coeff[13]*other.coeff[3] -
            2*self.coeff[24]*other.coeff[11] -
            2*self.coeff[21]*other.coeff[8] - self.coeff[19]*other.coeff[7]
            + 2*self.coeff[28]*other.coeff[17] +
            self.coeff[4]*other.coeff[0] - self.coeff[11]*other.coeff[2] +
            self.coeff[1]*other.coeff[8] + 2*self.coeff[31]*other.coeff[26]
            - self.coeff[16]*other.coeff[26] -
            self.coeff[10]*other.coeff[22] + self.coeff[3]*other.coeff[13],
    -self.coeff[23]*other.coeff[10] + 2*self.coeff[23]*other.coeff[30] +
            2*self.coeff[12]*other.coeff[24] + self.coeff[1]*other.coeff[9] -
            self.coeff[16]*other.coeff[27] - self.coeff[18]*other.coeff[6] -
            self.coeff[10]*other.coeff[23] + self.coeff[27]*other.coeff[16] +
            2*self.coeff[14]*other.coeff[25] - self.coeff[14]*other.coeff[3] +
            self.coeff[3]*other.coeff[14] + 2*self.coeff[18]*other.coeff[28] -
            self.coeff[9]*other.coeff[1] + 2*self.coeff[9]*other.coeff[21] -
            self.coeff[7]*other.coeff[20] - 2*self.coeff[5]*other.coeff[15] +
            other.coeff[5]*self.coeff[0] + self.coeff[2]*other.coeff[12] -
            self.coeff[6]*other.coeff[18] - 2*self.coeff[27]*other.coeff[31] -
            self.coeff[20]*other.coeff[7] + 2*self.coeff[20]*other.coeff[29] -
            self.coeff[12]*other.coeff[2] + self.coeff[5]*other.coeff[0],
    -2*self.coeff[12]*other.coeff[8] - 2*self.coeff[18]*other.coeff[4] +
            2*self.coeff[9]*other.coeff[11] + self.coeff[1]*other.coeff[2] +
            self.coeff[16]*other.coeff[3] + self.coeff[10]*other.coeff[7] -
            self.coeff[7]*other.coeff[10] + 2*self.coeff[27]*other.coeff[13] +
            2*self.coeff[14]*other.coeff[26] + self.coeff[3]*other.coeff[16] -
            2*self.coeff[5]*other.coeff[17] + other.coeff[6]*self.coeff[0] -
            self.coeff[2]*other.coeff[1] + self.coeff[6]*other.coeff[0] +
            2*self.coeff[20]*other.coeff[22] - 2*self.coeff[23]*other.coeff[19],
    -self.coeff[3]*other.coeff[1] - self.coeff[2]*other.coeff[16] -
            2*self.coeff[18]*other.coeff[22] + 2*self.coeff[9]*other.coeff[13] -
            2*self.coeff[20]*other.coeff[4] + 2*self.coeff[23]*other.coeff[17] -
            2*self.coeff[12]*other.coeff[26] - 2*self.coeff[27]*other.coeff[11]-
            2*self.coeff[14]*other.coeff[8] - 2*self.coeff[5]*other.coeff[19] +
            other.coeff[7]*self.coeff[0] + self.coeff[1]*other.coeff[3] -
            self.coeff[16]*other.coeff[2] - self.coeff[10]*other.coeff[6] +
            self.coeff[6]*other.coeff[10] + self.coeff[7]*other.coeff[0],
    -self.coeff[17]*other.coeff[2] + 2*self.coeff[30]*other.coeff[26] +
            self.coeff[7]*other.coeff[13] - 2*self.coeff[15]*other.coeff[8] -
            2*self.coeff[29]*other.coeff[13] + self.coeff[8]*other.coeff[0] +
            self.coeff[6]*other.coeff[11] + self.coeff[22]*other.coeff[16] -
            self.coeff[26]*other.coeff[10] - self.coeff[13]*other.coeff[7] +
            2*self.coeff[25]*other.coeff[19] + 2*self.coeff[24]*other.coeff[17]-
            2*self.coeff[21]*other.coeff[4] - self.coeff[19]*other.coeff[3] -
            2*self.coeff[28]*other.coeff[11] - self.coeff[4]*other.coeff[1] -
            self.coeff[11]*other.coeff[6] + self.coeff[1]*other.coeff[4] +
            2*self.coeff[31]*other.coeff[22] - self.coeff[16]*other.coeff[22] -
            self.coeff[10]*other.coeff[26] - self.coeff[3]*other.coeff[19] -
            self.coeff[2]*other.coeff[17] + other.coeff[8]*self.coeff[0],
    2*self.coeff[20]*other.coeff[25] - self.coeff[12]*other.coeff[6] +
            self.coeff[23]*other.coeff[16] - 2*self.coeff[23]*other.coeff[31] +
            2*self.coeff[12]*other.coeff[28] + self.coeff[1]*other.coeff[5] -
            self.coeff[16]*other.coeff[23] - self.coeff[18]*other.coeff[2] +
            2*self.coeff[18]*other.coeff[24] - self.coeff[10]*other.coeff[27] -
            self.coeff[27]*other.coeff[10] - self.coeff[5]*other.coeff[1] -
            self.coeff[14]*other.coeff[7] - self.coeff[3]*other.coeff[20] -
            self.coeff[2]*other.coeff[18] + self.coeff[7]*other.coeff[14] +
            self.coeff[9]*other.coeff[0] - 2*self.coeff[9]*other.coeff[15] +
            other.coeff[9]*self.coeff[0] + 2*self.coeff[5]*other.coeff[21] +
            self.coeff[6]*other.coeff[12] + 2*self.coeff[27]*other.coeff[30] +
            2*self.coeff[14]*other.coeff[29] - self.coeff[20]*other.coeff[3],
    2*self.coeff[18]*other.coeff[19] - self.coeff[3]*other.coeff[2] +
            other.coeff[10]*self.coeff[0] + self.coeff[2]*other.coeff[3] +
            self.coeff[1]*other.coeff[16] + self.coeff[16]*other.coeff[1] +
            self.coeff[7]*other.coeff[6] + self.coeff[10]*other.coeff[0] -
            self.coeff[6]*other.coeff[7] + 2*self.coeff[9]*other.coeff[26] -
            2*self.coeff[20]*other.coeff[17] - 2*self.coeff[23]*other.coeff[4] +
            2*self.coeff[12]*other.coeff[13] - 2*self.coeff[5]*other.coeff[22] +
            2*self.coeff[27]*other.coeff[8] - 2*self.coeff[14]*other.coeff[11],
    -2*self.coeff[24]*other.coeff[4] - 2*self.coeff[21]*other.coeff[17] -
            self.coeff[19]*other.coeff[16] + 2*self.coeff[28]*other.coeff[8] -
            self.coeff[4]*other.coeff[2] + self.coeff[11]*other.coeff[0] +
            self.coeff[1]*other.coeff[17] - 2*self.coeff[31]*other.coeff[19] +
            self.coeff[16]*other.coeff[19] + self.coeff[10]*other.coeff[13] +
            self.coeff[17]*other.coeff[1] - 2*self.coeff[30]*other.coeff[13] +
            self.coeff[7]*other.coeff[26] + self.coeff[8]*other.coeff[6] -
            self.coeff[3]*other.coeff[22] + self.coeff[2]*other.coeff[4] +
            other.coeff[11]*self.coeff[0] - 2*self.coeff[15]*other.coeff[11] -
            2*self.coeff[29]*other.coeff[26] - self.coeff[6]*other.coeff[8] -
            self.coeff[22]*other.coeff[3] + self.coeff[26]*other.coeff[7] -
            self.coeff[13]*other.coeff[10] + 2*self.coeff[25]*other.coeff[22],
    self.coeff[27]*other.coeff[7] - self.coeff[5]*other.coeff[2] -
            self.coeff[14]*other.coeff[10] - self.coeff[3]*other.coeff[23] +
            self.coeff[2]*other.coeff[5] + other.coeff[12]*self.coeff[0] +
            2*self.coeff[5]*other.coeff[24] - self.coeff[6]*other.coeff[9] -
            2*self.coeff[27]*other.coeff[29] + 2*self.coeff[14]*other.coeff[30]-
            self.coeff[20]*other.coeff[16] + 2*self.coeff[20]*other.coeff[31] -
            self.coeff[23]*other.coeff[3] + 2*self.coeff[23]*other.coeff[25] +
            self.coeff[12]*other.coeff[0] - 2*self.coeff[12]*other.coeff[15] +
            self.coeff[1]*other.coeff[18] + self.coeff[18]*other.coeff[1] +
            self.coeff[16]*other.coeff[20] + self.coeff[10]*other.coeff[14] -
            2*self.coeff[18]*other.coeff[21] + self.coeff[9]*other.coeff[6] +
            self.coeff[7]*other.coeff[27] - 2*self.coeff[9]*other.coeff[28],
    self.coeff[2]*other.coeff[22] + other.coeff[13]*self.coeff[0] -
            2*self.coeff[15]*other.coeff[13] - self.coeff[6]*other.coeff[26] +
            2*self.coeff[29]*other.coeff[8] + self.coeff[22]*other.coeff[2] -
            self.coeff[26]*other.coeff[6] - 2*self.coeff[25]*other.coeff[4] +
            self.coeff[13]*other.coeff[0] - 2*self.coeff[24]*other.coeff[22] -
            2*self.coeff[21]*other.coeff[19] + self.coeff[19]*other.coeff[1] +
            2*self.coeff[28]*other.coeff[26] - self.coeff[4]*other.coeff[3] +
            self.coeff[11]*other.coeff[10] + self.coeff[1]*other.coeff[19] +
            2*self.coeff[31]*other.coeff[17] - self.coeff[16]*other.coeff[17] -
            self.coeff[10]*other.coeff[11] + 2*self.coeff[30]*other.coeff[11] +
            self.coeff[17]*other.coeff[16] - self.coeff[7]*other.coeff[8] +
            self.coeff[8]*other.coeff[7] + self.coeff[3]*other.coeff[4],
    self.coeff[9]*other.coeff[7] - 2*self.coeff[18]*other.coeff[31] -
            self.coeff[7]*other.coeff[9] - 2*self.coeff[9]*other.coeff[29] +
            self.coeff[3]*other.coeff[5] + self.coeff[2]*other.coeff[23] +
            other.coeff[14]*self.coeff[0] + 2*self.coeff[5]*other.coeff[25] -
            self.coeff[6]*other.coeff[27] + 2*self.coeff[27]*other.coeff[28] +
            self.coeff[14]*other.coeff[0] - 2*self.coeff[14]*other.coeff[15] +
            self.coeff[20]*other.coeff[1] - 2*self.coeff[20]*other.coeff[21] +
            self.coeff[12]*other.coeff[10] + self.coeff[23]*other.coeff[2] -
            2*self.coeff[23]*other.coeff[24] - 2*self.coeff[12]*other.coeff[30]+
            self.coeff[1]*other.coeff[20] + self.coeff[18]*other.coeff[16] -
            self.coeff[16]*other.coeff[18] - self.coeff[10]*other.coeff[12] -
            self.coeff[27]*other.coeff[6] - self.coeff[5]*other.coeff[3],
    -self.coeff[27]*other.coeff[26] + self.coeff[12]*other.coeff[11] +
            self.coeff[23]*other.coeff[22] - self.coeff[31]*other.coeff[16] -
            self.coeff[30]*other.coeff[10] - self.coeff[29]*other.coeff[7] +
            self.coeff[25]*other.coeff[3] + self.coeff[24]*other.coeff[2] +
            self.coeff[21]*other.coeff[1] - self.coeff[28]*other.coeff[6] +
            self.coeff[15]*other.coeff[0] + self.coeff[26]*other.coeff[27] -
            2*self.coeff[25]*other.coeff[25] - self.coeff[13]*other.coeff[14] -
            2*self.coeff[24]*other.coeff[24] - 2*self.coeff[21]*other.coeff[21]-
            self.coeff[19]*other.coeff[20] + 2*self.coeff[28]*other.coeff[28] +
            self.coeff[4]*other.coeff[5] + self.coeff[3]*other.coeff[25] +
            other.coeff[15]*self.coeff[0] + self.coeff[2]*other.coeff[24] -
            2*self.coeff[15]*other.coeff[15] - self.coeff[6]*other.coeff[28] +
            2*self.coeff[29]*other.coeff[29] - self.coeff[11]*other.coeff[12] +
            self.coeff[1]*other.coeff[21] + 2*self.coeff[31]*other.coeff[31] -
            self.coeff[16]*other.coeff[31] - self.coeff[10]*other.coeff[30] +
            2*self.coeff[30]*other.coeff[30] - self.coeff[17]*other.coeff[18] -
            self.coeff[8]*other.coeff[9] - self.coeff[7]*other.coeff[29] -
            self.coeff[22]*other.coeff[23] - self.coeff[5]*other.coeff[4] +
            self.coeff[20]*other.coeff[19] + self.coeff[18]*other.coeff[17] +
            self.coeff[9]*other.coeff[8] + self.coeff[14]*other.coeff[13],
    -2*self.coeff[27]*other.coeff[4] - 2*self.coeff[14]*other.coeff[17] +
            2*self.coeff[5]*other.coeff[26] + self.coeff[6]*other.coeff[3] +
            self.coeff[1]*other.coeff[10] + self.coeff[10]*other.coeff[1] +
            self.coeff[16]*other.coeff[0] - self.coeff[7]*other.coeff[2] -
            2*self.coeff[20]*other.coeff[11] + 2*self.coeff[23]*other.coeff[8] +
            2*self.coeff[12]*other.coeff[19] + 2*self.coeff[18]*other.coeff[13]-
            2*self.coeff[9]*other.coeff[22] + self.coeff[3]*other.coeff[6] +
            other.coeff[16]*self.coeff[0] - self.coeff[2]*other.coeff[7],
    2*self.coeff[29]*other.coeff[22] + self.coeff[6]*other.coeff[4] +
            self.coeff[22]*other.coeff[7] - self.coeff[26]*other.coeff[3] -
            2*self.coeff[25]*other.coeff[26] - self.coeff[13]*other.coeff[16] +
            2*self.coeff[24]*other.coeff[8] - 2*self.coeff[21]*other.coeff[11] -
            self.coeff[19]*other.coeff[10] - 2*self.coeff[28]*other.coeff[4] +
            self.coeff[4]*other.coeff[6] + self.coeff[11]*other.coeff[1] +
            self.coeff[1]*other.coeff[11] + self.coeff[16]*other.coeff[13] -
            2*self.coeff[31]*other.coeff[13] + self.coeff[10]*other.coeff[19] -
            2*self.coeff[30]*other.coeff[19] + self.coeff[17]*other.coeff[0] -
            self.coeff[8]*other.coeff[2] - self.coeff[7]*other.coeff[22] +
            self.coeff[3]*other.coeff[26] + other.coeff[17]*self.coeff[0] -
            self.coeff[2]*other.coeff[8] - 2*self.coeff[15]*other.coeff[17],
    self.coeff[6]*other.coeff[5] + 2*self.coeff[27]*other.coeff[25] -
            2*self.coeff[23]*other.coeff[29] + 2*self.coeff[14]*other.coeff[31]-
            self.coeff[20]*other.coeff[10] + 2*self.coeff[20]*other.coeff[30] +
            self.coeff[12]*other.coeff[1] + self.coeff[23]*other.coeff[7] -
            2*self.coeff[12]*other.coeff[21] + self.coeff[1]*other.coeff[12] +
            self.coeff[16]*other.coeff[14] - self.coeff[9]*other.coeff[2] +
            self.coeff[10]*other.coeff[20] - 2*self.coeff[18]*other.coeff[15] +
            self.coeff[18]*other.coeff[0] + 2*self.coeff[9]*other.coeff[24] -
            self.coeff[7]*other.coeff[23] - self.coeff[27]*other.coeff[3] +
            self.coeff[5]*other.coeff[6] - self.coeff[14]*other.coeff[16] +
            self.coeff[3]*other.coeff[27] - 2*self.coeff[5]*other.coeff[28] +
            other.coeff[18]*self.coeff[0] - self.coeff[2]*other.coeff[9],
    2*self.coeff[24]*other.coeff[26] - 2*self.coeff[21]*other.coeff[13] -
            2*self.coeff[28]*other.coeff[22] + self.coeff[4]*other.coeff[7] +
            self.coeff[19]*other.coeff[0] + self.coeff[11]*other.coeff[16] +
            self.coeff[1]*other.coeff[13] + 2*self.coeff[31]*other.coeff[11] -
            self.coeff[16]*other.coeff[11] - self.coeff[10]*other.coeff[17] +
            self.coeff[17]*other.coeff[10] + 2*self.coeff[30]*other.coeff[17] +
            self.coeff[7]*other.coeff[4] - self.coeff[8]*other.coeff[3] -
            self.coeff[3]*other.coeff[8] + other.coeff[19]*self.coeff[0] -
            self.coeff[2]*other.coeff[26] - 2*self.coeff[15]*other.coeff[19] +
            self.coeff[6]*other.coeff[22] - 2*self.coeff[29]*other.coeff[4] -
            self.coeff[22]*other.coeff[6] + self.coeff[26]*other.coeff[2] +
            2*self.coeff[25]*other.coeff[8] + self.coeff[13]*other.coeff[1],
    self.coeff[27]*other.coeff[2] + self.coeff[5]*other.coeff[7] -
            2*self.coeff[14]*other.coeff[21] + self.coeff[14]*other.coeff[1] -
            self.coeff[3]*other.coeff[9] - 2*self.coeff[5]*other.coeff[29] +
            other.coeff[20]*self.coeff[0] - self.coeff[2]*other.coeff[27] +
            self.coeff[6]*other.coeff[23] + 2*self.coeff[23]*other.coeff[28] -
            2*self.coeff[27]*other.coeff[24] + self.coeff[20]*other.coeff[0] -
            2*self.coeff[20]*other.coeff[15] + self.coeff[12]*other.coeff[16] -
            self.coeff[23]*other.coeff[6] - 2*self.coeff[12]*other.coeff[31] +
            self.coeff[1]*other.coeff[14] - self.coeff[16]*other.coeff[12] +
            self.coeff[18]*other.coeff[10] - self.coeff[9]*other.coeff[3] -
            self.coeff[10]*other.coeff[18] - 2*self.coeff[18]*other.coeff[30] +
            2*self.coeff[9]*other.coeff[25] + self.coeff[7]*other.coeff[5],
    self.coeff[27]*other.coeff[22] + self.coeff[20]*other.coeff[13] -
            self.coeff[23]*other.coeff[26] + self.coeff[12]*other.coeff[17] +
            self.coeff[14]*other.coeff[19] + self.coeff[5]*other.coeff[8] +
            self.coeff[18]*other.coeff[11] - self.coeff[9]*other.coeff[4] -
            self.coeff[26]*other.coeff[23] + 2*self.coeff[25]*other.coeff[29] -
            self.coeff[13]*other.coeff[20] + 2*self.coeff[24]*other.coeff[28] -
            2*self.coeff[21]*other.coeff[15] - 2*self.coeff[28]*other.coeff[24]-
            self.coeff[4]*other.coeff[9] - self.coeff[19]*other.coeff[14] +
            self.coeff[1]*other.coeff[15] + 2*self.coeff[31]*other.coeff[30] -
            self.coeff[16]*other.coeff[30] - self.coeff[10]*other.coeff[31] -
            self.coeff[17]*other.coeff[12] + 2*self.coeff[30]*other.coeff[31] +
            self.coeff[7]*other.coeff[25] + self.coeff[8]*other.coeff[5] -
            self.coeff[11]*other.coeff[18] - self.coeff[3]*other.coeff[29] -
            self.coeff[2]*other.coeff[28] + other.coeff[21]*self.coeff[0] -
            2*self.coeff[15]*other.coeff[21] + self.coeff[6]*other.coeff[24] -
            2*self.coeff[29]*other.coeff[25] + self.coeff[22]*other.coeff[27] -
            self.coeff[31]*other.coeff[10] - self.coeff[30]*other.coeff[16] +
            self.coeff[15]*other.coeff[1] - self.coeff[25]*other.coeff[7] -
            self.coeff[24]*other.coeff[6] + self.coeff[21]*other.coeff[0] +
            self.coeff[28]*other.coeff[2] + self.coeff[29]*other.coeff[3],
    self.coeff[16]*other.coeff[8] + self.coeff[10]*other.coeff[4] -
            2*self.coeff[30]*other.coeff[4] - self.coeff[17]*other.coeff[7] +
            self.coeff[7]*other.coeff[17] - self.coeff[8]*other.coeff[16] -
            self.coeff[3]*other.coeff[11] + other.coeff[22]*self.coeff[0] +
            self.coeff[2]*other.coeff[13] - 2*self.coeff[15]*other.coeff[22] -
            self.coeff[6]*other.coeff[19] - 2*self.coeff[29]*other.coeff[17] +
            self.coeff[22]*other.coeff[0] - self.coeff[26]*other.coeff[1] +
            2*self.coeff[25]*other.coeff[11] + self.coeff[13]*other.coeff[2] -
            2*self.coeff[24]*other.coeff[13] - 2*self.coeff[21]*other.coeff[26]+
            self.coeff[19]*other.coeff[6] + 2*self.coeff[28]*other.coeff[19] +
            self.coeff[4]*other.coeff[10] - self.coeff[11]*other.coeff[3] +
            self.coeff[1]*other.coeff[26] - 2*self.coeff[31]*other.coeff[8],
    -self.coeff[9]*other.coeff[16] + 2*self.coeff[18]*other.coeff[29] +
            2*self.coeff[9]*other.coeff[31] + self.coeff[7]*other.coeff[18] -
            self.coeff[27]*other.coeff[1] + self.coeff[5]*other.coeff[10] -
            2*self.coeff[14]*other.coeff[24] + self.coeff[14]*other.coeff[2] -
            self.coeff[3]*other.coeff[12] - 2*self.coeff[5]*other.coeff[30] +
            other.coeff[23]*self.coeff[0] + self.coeff[2]*other.coeff[14] -
            self.coeff[6]*other.coeff[20] + self.coeff[23]*other.coeff[0] -
            2*self.coeff[23]*other.coeff[15] + 2*self.coeff[27]*other.coeff[21]+
            self.coeff[20]*other.coeff[6] - 2*self.coeff[20]*other.coeff[28] -
            self.coeff[12]*other.coeff[3] + 2*self.coeff[12]*other.coeff[25] +
            self.coeff[1]*other.coeff[27] + self.coeff[16]*other.coeff[9] -
            self.coeff[18]*other.coeff[7] + self.coeff[10]*other.coeff[5],
    self.coeff[30]*other.coeff[3] + self.coeff[15]*other.coeff[2] +
            self.coeff[29]*other.coeff[16] - 2*self.coeff[31]*other.coeff[29] +
            self.coeff[16]*other.coeff[29] + self.coeff[10]*other.coeff[25] -
            2*self.coeff[30]*other.coeff[25] + self.coeff[17]*other.coeff[9] +
            self.coeff[7]*other.coeff[31] - self.coeff[3]*other.coeff[30] +
            self.coeff[2]*other.coeff[15] + other.coeff[24]*self.coeff[0] -
            2*self.coeff[15]*other.coeff[24] - 2*self.coeff[29]*other.coeff[31]-
            self.coeff[6]*other.coeff[21] + self.coeff[8]*other.coeff[18] -
            self.coeff[22]*other.coeff[14] + self.coeff[26]*other.coeff[20] +
            2*self.coeff[25]*other.coeff[30] - self.coeff[25]*other.coeff[10] +
            self.coeff[24]*other.coeff[0] + self.coeff[21]*other.coeff[6] -
            self.coeff[28]*other.coeff[1] + self.coeff[31]*other.coeff[7] -
            self.coeff[13]*other.coeff[23] - 2*self.coeff[24]*other.coeff[15] -
            2*self.coeff[21]*other.coeff[28] - self.coeff[19]*other.coeff[27] +
            2*self.coeff[28]*other.coeff[21] - self.coeff[4]*other.coeff[12] +
            self.coeff[11]*other.coeff[5] + self.coeff[1]*other.coeff[28] +
            self.coeff[14]*other.coeff[22] + self.coeff[5]*other.coeff[11] -
            self.coeff[27]*other.coeff[19] + self.coeff[20]*other.coeff[26] +
            self.coeff[23]*other.coeff[13] - self.coeff[12]*other.coeff[4] -
            self.coeff[18]*other.coeff[8] - self.coeff[9]*other.coeff[17],
    -self.coeff[7]*other.coeff[21] + self.coeff[3]*other.coeff[15] +
            self.coeff[2]*other.coeff[30] + other.coeff[25]*self.coeff[0] -
            2*self.coeff[15]*other.coeff[25] - self.coeff[6]*other.coeff[31] -
            2*self.coeff[21]*other.coeff[29] + 2*self.coeff[28]*other.coeff[31]+
            self.coeff[19]*other.coeff[9] - self.coeff[4]*other.coeff[14] +
            2*self.coeff[29]*other.coeff[21] + self.coeff[8]*other.coeff[20] +
            self.coeff[22]*other.coeff[12] - self.coeff[26]*other.coeff[18] +
            self.coeff[13]*other.coeff[5] - 2*self.coeff[25]*other.coeff[15] -
            2*self.coeff[24]*other.coeff[30] + self.coeff[11]*other.coeff[23] +
            self.coeff[1]*other.coeff[29] + 2*self.coeff[31]*other.coeff[28] -
            self.coeff[16]*other.coeff[28] - self.coeff[10]*other.coeff[24] +
            2*self.coeff[30]*other.coeff[24] + self.coeff[17]*other.coeff[27] -
            self.coeff[31]*other.coeff[6] - self.coeff[30]*other.coeff[2] +
            self.coeff[15]*other.coeff[3] - self.coeff[28]*other.coeff[16] -
            self.coeff[29]*other.coeff[1] + self.coeff[25]*other.coeff[0] +
            self.coeff[24]*other.coeff[10] + self.coeff[21]*other.coeff[7] -
            self.coeff[14]*other.coeff[4] + self.coeff[5]*other.coeff[13] +
            self.coeff[27]*other.coeff[17] - self.coeff[20]*other.coeff[8] -
            self.coeff[23]*other.coeff[11] - self.coeff[12]*other.coeff[22] -
            self.coeff[18]*other.coeff[26] - self.coeff[9]*other.coeff[19],
    self.coeff[3]*other.coeff[17] - self.coeff[2]*other.coeff[19] +
            other.coeff[26]*self.coeff[0] - 2*self.coeff[15]*other.coeff[26] +
            2*self.coeff[29]*other.coeff[11] + self.coeff[6]*other.coeff[13] -
            self.coeff[22]*other.coeff[1] + self.coeff[26]*other.coeff[0] +
            self.coeff[13]*other.coeff[6] - 2*self.coeff[25]*other.coeff[17] +
            2*self.coeff[24]*other.coeff[19] - 2*self.coeff[21]*other.coeff[22]+
            self.coeff[19]*other.coeff[2] - 2*self.coeff[28]*other.coeff[13] -
            self.coeff[4]*other.coeff[16] - self.coeff[11]*other.coeff[7] +
            self.coeff[1]*other.coeff[22] - 2*self.coeff[31]*other.coeff[4] +
            self.coeff[16]*other.coeff[4] + self.coeff[10]*other.coeff[8] -
            2*self.coeff[30]*other.coeff[8] - self.coeff[17]*other.coeff[3] -
            self.coeff[7]*other.coeff[11] + self.coeff[8]*other.coeff[10],
    self.coeff[27]*other.coeff[0] - 2*self.coeff[27]*other.coeff[15] -
            2*self.coeff[14]*other.coeff[28] + self.coeff[20]*other.coeff[2] -
            2*self.coeff[20]*other.coeff[24] - self.coeff[12]*other.coeff[7] -
            self.coeff[23]*other.coeff[1] + 2*self.coeff[23]*other.coeff[21] +
            2*self.coeff[12]*other.coeff[29] + self.coeff[1]*other.coeff[23] +
            self.coeff[16]*other.coeff[5] - self.coeff[18]*other.coeff[3] +
            2*self.coeff[18]*other.coeff[25] + self.coeff[10]*other.coeff[9] +
            self.coeff[9]*other.coeff[10] - self.coeff[7]*other.coeff[12] -
            2*self.coeff[9]*other.coeff[30] - self.coeff[5]*other.coeff[16] +
            self.coeff[14]*other.coeff[6] + self.coeff[3]*other.coeff[18] -
            self.coeff[2]*other.coeff[20] + 2*self.coeff[5]*other.coeff[31] +
            other.coeff[27]*self.coeff[0] + self.coeff[6]*other.coeff[14],
    self.coeff[3]*other.coeff[31] + other.coeff[28]*self.coeff[0] -
            self.coeff[2]*other.coeff[21] - 2*self.coeff[15]*other.coeff[28] +
            self.coeff[6]*other.coeff[15] - self.coeff[8]*other.coeff[12] +
            self.coeff[15]*other.coeff[6] - self.coeff[7]*other.coeff[30] +
            self.coeff[10]*other.coeff[29] + self.coeff[17]*other.coeff[5] -
            2*self.coeff[30]*other.coeff[29] + self.coeff[16]*other.coeff[25] +
            self.coeff[30]*other.coeff[7] + 2*self.coeff[29]*other.coeff[30] -
            self.coeff[29]*other.coeff[10] + self.coeff[22]*other.coeff[20] -
            self.coeff[26]*other.coeff[14] + self.coeff[25]*other.coeff[16] -
            2*self.coeff[25]*other.coeff[31] - self.coeff[13]*other.coeff[27] -
            self.coeff[24]*other.coeff[1] + 2*self.coeff[24]*other.coeff[21] +
            self.coeff[21]*other.coeff[2] - 2*self.coeff[21]*other.coeff[24] -
            self.coeff[19]*other.coeff[23] + self.coeff[28]*other.coeff[0] -
            2*self.coeff[28]*other.coeff[15] + self.coeff[4]*other.coeff[18] +
            self.coeff[11]*other.coeff[9] + self.coeff[1]*other.coeff[24] +
            self.coeff[31]*other.coeff[3] - 2*self.coeff[31]*other.coeff[25] -
            self.coeff[12]*other.coeff[8] - self.coeff[18]*other.coeff[4] +
            self.coeff[9]*other.coeff[11] + self.coeff[27]*other.coeff[13] +
            self.coeff[14]*other.coeff[26] - self.coeff[5]*other.coeff[17] +
            self.coeff[20]*other.coeff[22] - self.coeff[23]*other.coeff[19],
    -self.coeff[31]*other.coeff[2] + 2*self.coeff[31]*other.coeff[24] -
            self.coeff[16]*other.coeff[24] - self.coeff[18]*other.coeff[22] -
            self.coeff[30]*other.coeff[6] - self.coeff[10]*other.coeff[28] +
            2*self.coeff[30]*other.coeff[28] + self.coeff[17]*other.coeff[23] +
            self.coeff[9]*other.coeff[13] + self.coeff[7]*other.coeff[15] +
            self.coeff[15]*other.coeff[7] - self.coeff[25]*other.coeff[1] +
            2*self.coeff[25]*other.coeff[21] + self.coeff[13]*other.coeff[9] -
            self.coeff[24]*other.coeff[16] + 2*self.coeff[24]*other.coeff[31] +
            self.coeff[21]*other.coeff[3] - 2*self.coeff[21]*other.coeff[25] -
            self.coeff[20]*other.coeff[4] + self.coeff[28]*other.coeff[10] +
            self.coeff[19]*other.coeff[5] - 2*self.coeff[28]*other.coeff[30] +
            self.coeff[4]*other.coeff[20] + self.coeff[23]*other.coeff[17] -
            self.coeff[12]*other.coeff[26] + self.coeff[11]*other.coeff[27] +
            self.coeff[1]*other.coeff[25] - self.coeff[27]*other.coeff[11] -
            self.coeff[14]*other.coeff[8] - self.coeff[3]*other.coeff[21] -
            self.coeff[2]*other.coeff[31] + other.coeff[29]*self.coeff[0] -
            self.coeff[5]*other.coeff[19] - 2*self.coeff[15]*other.coeff[29] +
            self.coeff[6]*other.coeff[30] + self.coeff[29]*other.coeff[0] -
            2*self.coeff[29]*other.coeff[15] - self.coeff[8]*other.coeff[14] -
            self.coeff[22]*other.coeff[18] + self.coeff[26]*other.coeff[12],
    self.coeff[16]*other.coeff[21] - self.coeff[8]*other.coeff[27] +
            self.coeff[10]*other.coeff[15] - 2*self.coeff[31]*other.coeff[21] -
            2*self.coeff[30]*other.coeff[15] + self.coeff[1]*other.coeff[31] +
            self.coeff[15]*other.coeff[10] + self.coeff[18]*other.coeff[19] +
            self.coeff[31]*other.coeff[1] + self.coeff[7]*other.coeff[28] +
            self.coeff[9]*other.coeff[26] + self.coeff[30]*other.coeff[0] -
            self.coeff[17]*other.coeff[20] - self.coeff[20]*other.coeff[17] -
            self.coeff[28]*other.coeff[7] + 2*self.coeff[28]*other.coeff[29] +
            self.coeff[4]*other.coeff[23] - self.coeff[23]*other.coeff[4] +
            self.coeff[19]*other.coeff[18] + self.coeff[12]*other.coeff[13] -
            self.coeff[11]*other.coeff[14] - self.coeff[5]*other.coeff[22] -
            2*self.coeff[15]*other.coeff[30] - self.coeff[6]*other.coeff[29] -
            2*self.coeff[29]*other.coeff[28] + self.coeff[29]*other.coeff[6] +
            self.coeff[27]*other.coeff[8] + self.coeff[22]*other.coeff[5] -
            self.coeff[26]*other.coeff[9] - self.coeff[25]*other.coeff[2] +
            2*self.coeff[25]*other.coeff[24] + self.coeff[13]*other.coeff[12] +
            self.coeff[24]*other.coeff[3] - 2*self.coeff[24]*other.coeff[25] +
            self.coeff[21]*other.coeff[16] - 2*self.coeff[21]*other.coeff[31] -
            self.coeff[14]*other.coeff[11] - self.coeff[3]*other.coeff[24] +
            other.coeff[30]*self.coeff[0] + self.coeff[2]*other.coeff[25],
    self.coeff[6]*other.coeff[25] + 2*self.coeff[29]*other.coeff[24] -
            self.coeff[29]*other.coeff[2] + self.coeff[8]*other.coeff[23] -
            self.coeff[27]*other.coeff[4] - self.coeff[22]*other.coeff[9] +
            self.coeff[26]*other.coeff[5] + self.coeff[25]*other.coeff[6] -
            2*self.coeff[25]*other.coeff[28] - self.coeff[14]*other.coeff[17] +
            self.coeff[3]*other.coeff[28] - self.coeff[2]*other.coeff[29] +
            other.coeff[31]*self.coeff[0] - 2*self.coeff[15]*other.coeff[31] +
            self.coeff[5]*other.coeff[26] + self.coeff[15]*other.coeff[16] -
            self.coeff[7]*other.coeff[24] - self.coeff[17]*other.coeff[14] +
            self.coeff[13]*other.coeff[18] - self.coeff[24]*other.coeff[7] +
            2*self.coeff[24]*other.coeff[29] + self.coeff[21]*other.coeff[10] -
            self.coeff[20]*other.coeff[11] - 2*self.coeff[21]*other.coeff[30] +
            self.coeff[28]*other.coeff[3] + self.coeff[19]*other.coeff[12] -
            2*self.coeff[28]*other.coeff[25] - self.coeff[4]*other.coeff[27] +
            self.coeff[23]*other.coeff[8] + self.coeff[12]*other.coeff[19] -
            self.coeff[11]*other.coeff[20] + self.coeff[1]*other.coeff[30] +
            self.coeff[31]*other.coeff[0] - 2*self.coeff[31]*other.coeff[15] +
            self.coeff[16]*other.coeff[15] + self.coeff[10]*other.coeff[21] +
            self.coeff[18]*other.coeff[13] + self.coeff[30]*other.coeff[1] -
            2*self.coeff[30]*other.coeff[21] - self.coeff[9]*other.coeff[22]
            ]
            return cga_object(out)
        except:
            cof = np.zeros(self.dim)
            for i in range(self.dim):
                cof[i] = other*self.coeff[i]
            return cga_object(cof)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """division by non cga_objects

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other,cga_object):
            print("Division of cga_objects not allowed")
        else:
            # cof = np.zeros(self.dim)
            # for i in range(self.dim):
                # cof[i] = self.coeff[i]/other
            # return cga_object(cof)
            return (1/other)*self

    __rtruediv__ = __truediv__

    def __xor__(self, other):
        """outer / wedge product

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        out =[
        self.coeff[0]*other.coeff[0] + self.coeff[4]*other.coeff[5] -
            self.coeff[5]*other.coeff[4] - self.coeff[15]*other.coeff[15],
 self.coeff[0]*other.coeff[1] + self.coeff[1]*other.coeff[0] -
            self.coeff[4]*other.coeff[9] + self.coeff[5]*other.coeff[8] +
            self.coeff[8]*other.coeff[5] - self.coeff[9]*other.coeff[4] -
            self.coeff[15]*other.coeff[21] - self.coeff[21]*other.coeff[15],
 self.coeff[0]*other.coeff[2] + self.coeff[2]*other.coeff[0] -
            self.coeff[4]*other.coeff[12] + self.coeff[5]*other.coeff[11] +
            self.coeff[11]*other.coeff[5] - self.coeff[12]*other.coeff[4] -
            self.coeff[15]*other.coeff[24] - self.coeff[24]*other.coeff[15],
 self.coeff[0]*other.coeff[3] + self.coeff[3]*other.coeff[0] -
            self.coeff[4]*other.coeff[14] + self.coeff[5]*other.coeff[13] +
            self.coeff[13]*other.coeff[5] - self.coeff[14]*other.coeff[4] -
            self.coeff[15]*other.coeff[25] - self.coeff[25]*other.coeff[15],
 self.coeff[0]*other.coeff[4] + self.coeff[4]*other.coeff[0] -
            self.coeff[4]*other.coeff[15] - self.coeff[15]*other.coeff[4],
 self.coeff[0]*other.coeff[5] + self.coeff[5]*other.coeff[0] -
            self.coeff[5]*other.coeff[15] - self.coeff[15]*other.coeff[5],
 -self.coeff[2]*other.coeff[1] + self.coeff[1]*other.coeff[2] +
            other.coeff[6]*self.coeff[0] - self.coeff[12]*other.coeff[8] -
            self.coeff[8]*other.coeff[12] - self.coeff[28]*other.coeff[15] -
            self.coeff[15]*other.coeff[28] - self.coeff[21]*other.coeff[24] +
            self.coeff[11]*other.coeff[9] - self.coeff[5]*other.coeff[17] +
            self.coeff[4]*other.coeff[18] + self.coeff[6]*other.coeff[0] +
            self.coeff[24]*other.coeff[21] - self.coeff[18]*other.coeff[4] +
            self.coeff[17]*other.coeff[5] + self.coeff[9]*other.coeff[11],
 -self.coeff[3]*other.coeff[1] + self.coeff[1]*other.coeff[3] +
            other.coeff[7]*self.coeff[0] - self.coeff[29]*other.coeff[15] -
            self.coeff[8]*other.coeff[14] + self.coeff[13]*other.coeff[9] -
            self.coeff[15]*other.coeff[29] - self.coeff[21]*other.coeff[25] -
            self.coeff[5]*other.coeff[19] + self.coeff[4]*other.coeff[20] +
            self.coeff[7]*other.coeff[0] + self.coeff[25]*other.coeff[21] -
            self.coeff[20]*other.coeff[4] + self.coeff[19]*other.coeff[5] +
            self.coeff[9]*other.coeff[13] - self.coeff[14]*other.coeff[8],
 self.coeff[0]*other.coeff[8] + self.coeff[1]*other.coeff[4] -
            self.coeff[4]*other.coeff[1] + self.coeff[4]*other.coeff[21] +
            self.coeff[8]*other.coeff[0] - self.coeff[8]*other.coeff[15] -
            self.coeff[15]*other.coeff[8] - self.coeff[21]*other.coeff[4],
 self.coeff[0]*other.coeff[9] + self.coeff[1]*other.coeff[5] -
            self.coeff[5]*other.coeff[1] + self.coeff[5]*other.coeff[21] +
            self.coeff[9]*other.coeff[0] - self.coeff[9]*other.coeff[15] -
            self.coeff[15]*other.coeff[9] - self.coeff[21]*other.coeff[5],
 -self.coeff[3]*other.coeff[2] + self.coeff[2]*other.coeff[3] +
             other.coeff[10]*self.coeff[0] - self.coeff[30]*other.coeff[15] +
             self.coeff[12]*other.coeff[13] + self.coeff[13]*other.coeff[12] -
             self.coeff[15]*other.coeff[30] - self.coeff[24]*other.coeff[25] -
             self.coeff[11]*other.coeff[14] - self.coeff[5]*other.coeff[22] +
             self.coeff[4]*other.coeff[23] + self.coeff[10]*other.coeff[0] +
             self.coeff[25]*other.coeff[24] - self.coeff[14]*other.coeff[11] +
             self.coeff[22]*other.coeff[5] - self.coeff[23]*other.coeff[4],
 self.coeff[0]*other.coeff[11] + self.coeff[2]*other.coeff[4] -
             self.coeff[4]*other.coeff[2] + self.coeff[4]*other.coeff[24] +
             self.coeff[11]*other.coeff[0] - self.coeff[11]*other.coeff[15] -
             self.coeff[15]*other.coeff[11] - self.coeff[24]*other.coeff[4],
 self.coeff[0]*other.coeff[12] + self.coeff[2]*other.coeff[5] -
             self.coeff[5]*other.coeff[2] + self.coeff[5]*other.coeff[24] +
             self.coeff[12]*other.coeff[0] - self.coeff[12]*other.coeff[15] -
             self.coeff[15]*other.coeff[12] - self.coeff[24]*other.coeff[5],
             self.coeff[0]*other.coeff[13] + self.coeff[3]*other.coeff[4] -
             self.coeff[4]*other.coeff[3] + self.coeff[4]*other.coeff[25] +
             self.coeff[13]*other.coeff[0] - self.coeff[13]*other.coeff[15] -
             self.coeff[15]*other.coeff[13] - self.coeff[25]*other.coeff[4],
 self.coeff[0]*other.coeff[14] + self.coeff[3]*other.coeff[5] -
             self.coeff[5]*other.coeff[3] + self.coeff[5]*other.coeff[25] +
             self.coeff[14]*other.coeff[0] - self.coeff[14]*other.coeff[15] -
             self.coeff[15]*other.coeff[14] - self.coeff[25]*other.coeff[5],
 self.coeff[0]*other.coeff[15] + self.coeff[4]*other.coeff[5] -
             self.coeff[5]*other.coeff[4] + self.coeff[15]*other.coeff[0] -
             2*self.coeff[15]*other.coeff[15],
 self.coeff[16]*other.coeff[0] + self.coeff[3]*other.coeff[6] -
             self.coeff[7]*other.coeff[2] + self.coeff[24]*other.coeff[29] +
             self.coeff[5]*other.coeff[26] + self.coeff[13]*other.coeff[18] +
             self.coeff[12]*other.coeff[19] - self.coeff[11]*other.coeff[20] -
             self.coeff[4]*other.coeff[27] - self.coeff[31]*other.coeff[15] -
             self.coeff[21]*other.coeff[30] - self.coeff[20]*other.coeff[11] +
             self.coeff[19]*other.coeff[12] - self.coeff[30]*other.coeff[21] +
             self.coeff[18]*other.coeff[13] + self.coeff[29]*other.coeff[24] -
             self.coeff[17]*other.coeff[14] - self.coeff[28]*other.coeff[25] -
             self.coeff[15]*other.coeff[31] - self.coeff[9]*other.coeff[22] +
             self.coeff[8]*other.coeff[23] - self.coeff[14]*other.coeff[17] -
             self.coeff[25]*other.coeff[28] - self.coeff[22]*other.coeff[9] +
             self.coeff[23]*other.coeff[8] + self.coeff[26]*other.coeff[5] -
             self.coeff[27]*other.coeff[4] + self.coeff[6]*other.coeff[3] +
             other.coeff[16]*self.coeff[0] + self.coeff[10]*other.coeff[1] +
             self.coeff[1]*other.coeff[10] - self.coeff[2]*other.coeff[7],
 self.coeff[0]*other.coeff[17] + self.coeff[1]*other.coeff[11] -
             self.coeff[2]*other.coeff[8] + self.coeff[4]*other.coeff[6] -
             self.coeff[4]*other.coeff[28] + self.coeff[6]*other.coeff[4] -
             self.coeff[8]*other.coeff[2] + self.coeff[8]*other.coeff[24] +
             self.coeff[11]*other.coeff[1] - self.coeff[11]*other.coeff[21] -
             self.coeff[15]*other.coeff[17] + self.coeff[17]*other.coeff[0] -
             self.coeff[17]*other.coeff[15] - self.coeff[21]*other.coeff[11] +
             self.coeff[24]*other.coeff[8] - self.coeff[28]*other.coeff[4],
 self.coeff[0]*other.coeff[18] + self.coeff[1]*other.coeff[12] -
             self.coeff[2]*other.coeff[9] + self.coeff[5]*other.coeff[6] -
             self.coeff[5]*other.coeff[28] + self.coeff[6]*other.coeff[5] -
             self.coeff[9]*other.coeff[2] + self.coeff[9]*other.coeff[24] +
             self.coeff[12]*other.coeff[1] - self.coeff[12]*other.coeff[21] -
             self.coeff[15]*other.coeff[18] + self.coeff[18]*other.coeff[0] -
             self.coeff[18]*other.coeff[15] - self.coeff[21]*other.coeff[12] +
             self.coeff[24]*other.coeff[9] - self.coeff[28]*other.coeff[5],
 self.coeff[0]*other.coeff[19] + self.coeff[1]*other.coeff[13] -
             self.coeff[3]*other.coeff[8] + self.coeff[4]*other.coeff[7] -
             self.coeff[4]*other.coeff[29] + self.coeff[7]*other.coeff[4] -
             self.coeff[8]*other.coeff[3] + self.coeff[8]*other.coeff[25] +
             self.coeff[13]*other.coeff[1] - self.coeff[13]*other.coeff[21] -
             self.coeff[15]*other.coeff[19] + self.coeff[19]*other.coeff[0] -
             self.coeff[19]*other.coeff[15] - self.coeff[21]*other.coeff[13] +
             self.coeff[25]*other.coeff[8] - self.coeff[29]*other.coeff[4],
 self.coeff[0]*other.coeff[20] + self.coeff[1]*other.coeff[14] -
             self.coeff[3]*other.coeff[9] + self.coeff[5]*other.coeff[7] -
             self.coeff[5]*other.coeff[29] + self.coeff[7]*other.coeff[5] -
             self.coeff[9]*other.coeff[3] + self.coeff[9]*other.coeff[25] +
             self.coeff[14]*other.coeff[1] - self.coeff[14]*other.coeff[21] -
             self.coeff[15]*other.coeff[20] + self.coeff[20]*other.coeff[0] -
             self.coeff[20]*other.coeff[15] - self.coeff[21]*other.coeff[14] +
             self.coeff[25]*other.coeff[9] - self.coeff[29]*other.coeff[5],
 self.coeff[0]*other.coeff[21] + self.coeff[1]*other.coeff[15] -
             self.coeff[4]*other.coeff[9] + self.coeff[5]*other.coeff[8] +
             self.coeff[8]*other.coeff[5] - self.coeff[9]*other.coeff[4] +
             self.coeff[15]*other.coeff[1] - 2*self.coeff[15]*other.coeff[21] +
             self.coeff[21]*other.coeff[0] - 2*self.coeff[21]*other.coeff[15],
 self.coeff[0]*other.coeff[22] + self.coeff[2]*other.coeff[13] -
             self.coeff[3]*other.coeff[11] + self.coeff[4]*other.coeff[10] -
             self.coeff[4]*other.coeff[30] + self.coeff[10]*other.coeff[4] -
             self.coeff[11]*other.coeff[3] + self.coeff[11]*other.coeff[25] +
             self.coeff[13]*other.coeff[2] - self.coeff[13]*other.coeff[24] -
             self.coeff[15]*other.coeff[22] + self.coeff[22]*other.coeff[0] -
             self.coeff[22]*other.coeff[15] - self.coeff[24]*other.coeff[13] +
             self.coeff[25]*other.coeff[11] - self.coeff[30]*other.coeff[4],
 self.coeff[0]*other.coeff[23] + self.coeff[2]*other.coeff[14] -
             self.coeff[3]*other.coeff[12] + self.coeff[5]*other.coeff[10] -
             self.coeff[5]*other.coeff[30] + self.coeff[10]*other.coeff[5] -
             self.coeff[12]*other.coeff[3] + self.coeff[12]*other.coeff[25] +
             self.coeff[14]*other.coeff[2] - self.coeff[14]*other.coeff[24] -
             self.coeff[15]*other.coeff[23] + self.coeff[23]*other.coeff[0] -
             self.coeff[23]*other.coeff[15] - self.coeff[24]*other.coeff[14] +
             self.coeff[25]*other.coeff[12] - self.coeff[30]*other.coeff[5],
 self.coeff[0]*other.coeff[24] + self.coeff[2]*other.coeff[15] -
             self.coeff[4]*other.coeff[12] + self.coeff[5]*other.coeff[11] +
             self.coeff[11]*other.coeff[5] - self.coeff[12]*other.coeff[4] +
             self.coeff[15]*other.coeff[2] - 2*self.coeff[15]*other.coeff[24] +
             self.coeff[24]*other.coeff[0] - 2*self.coeff[24]*other.coeff[15],
 self.coeff[0]*other.coeff[25] + self.coeff[3]*other.coeff[15] -
             self.coeff[4]*other.coeff[14] + self.coeff[5]*other.coeff[13] +
             self.coeff[13]*other.coeff[5] - self.coeff[14]*other.coeff[4] +
             self.coeff[15]*other.coeff[3] - 2*self.coeff[15]*other.coeff[25] +
             self.coeff[25]*other.coeff[0] - 2*self.coeff[25]*other.coeff[15],
 self.coeff[6]*other.coeff[13] - self.coeff[13]*other.coeff[28] +
             self.coeff[13]*other.coeff[6] - self.coeff[26]*other.coeff[15] +
             self.coeff[26]*other.coeff[0] + other.coeff[26]*self.coeff[0] -
             self.coeff[15]*other.coeff[26] - self.coeff[11]*other.coeff[7] +
             self.coeff[11]*other.coeff[29] - self.coeff[4]*other.coeff[16] -
             self.coeff[7]*other.coeff[11] - self.coeff[21]*other.coeff[22] +
             self.coeff[3]*other.coeff[17] - self.coeff[2]*other.coeff[19] -
             self.coeff[31]*other.coeff[4] + self.coeff[1]*other.coeff[22] -
             self.coeff[30]*other.coeff[8] - self.coeff[19]*other.coeff[24] +
             self.coeff[19]*other.coeff[2] + self.coeff[29]*other.coeff[11] -
             self.coeff[28]*other.coeff[13] + self.coeff[17]*other.coeff[25] -
             self.coeff[17]*other.coeff[3] + self.coeff[8]*other.coeff[10] -
             self.coeff[8]*other.coeff[30] + self.coeff[24]*other.coeff[19] +
             self.coeff[10]*other.coeff[8] + self.coeff[16]*other.coeff[4] +
             self.coeff[22]*other.coeff[21] - self.coeff[22]*other.coeff[1] -
             self.coeff[25]*other.coeff[17] + self.coeff[4]*other.coeff[31],
 self.coeff[20]*other.coeff[2] + self.coeff[6]*other.coeff[14] -
             self.coeff[12]*other.coeff[7] - self.coeff[5]*other.coeff[16] -
             self.coeff[14]*other.coeff[28] - self.coeff[15]*other.coeff[27] -
             self.coeff[27]*other.coeff[15] + self.coeff[27]*other.coeff[0] +
             other.coeff[27]*self.coeff[0] + self.coeff[12]*other.coeff[29] +
             self.coeff[5]*other.coeff[31] - self.coeff[21]*other.coeff[23] +
             self.coeff[3]*other.coeff[18] - self.coeff[2]*other.coeff[20] -
             self.coeff[31]*other.coeff[5] + self.coeff[1]*other.coeff[23] -
             self.coeff[30]*other.coeff[9] - self.coeff[20]*other.coeff[24] +
             self.coeff[29]*other.coeff[12] - self.coeff[18]*other.coeff[3] +
             self.coeff[18]*other.coeff[25] - self.coeff[28]*other.coeff[14] +
             self.coeff[9]*other.coeff[10] + self.coeff[14]*other.coeff[6] -
             self.coeff[9]*other.coeff[30] - self.coeff[7]*other.coeff[12] +
             self.coeff[24]*other.coeff[20] + self.coeff[10]*other.coeff[9] +
             self.coeff[16]*other.coeff[5] - self.coeff[25]*other.coeff[18] +
             self.coeff[23]*other.coeff[21] - self.coeff[23]*other.coeff[1],
 2*self.coeff[24]*other.coeff[21] + self.coeff[6]*other.coeff[15] -
             self.coeff[12]*other.coeff[8] - self.coeff[8]*other.coeff[12] -
             2*self.coeff[28]*other.coeff[15] + other.coeff[28]*self.coeff[0] +
             self.coeff[28]*other.coeff[0] - 2*self.coeff[15]*other.coeff[28] +
             self.coeff[11]*other.coeff[9] - self.coeff[5]*other.coeff[17] +
             self.coeff[4]*other.coeff[18] - self.coeff[24]*other.coeff[1] -
             2*self.coeff[21]*other.coeff[24] - self.coeff[2]*other.coeff[21] +
             self.coeff[1]*other.coeff[24] - self.coeff[18]*other.coeff[4] +
             self.coeff[17]*other.coeff[5] + self.coeff[15]*other.coeff[6] +
             self.coeff[9]*other.coeff[11] + self.coeff[21]*other.coeff[2],
 -2*self.coeff[29]*other.coeff[15] - self.coeff[8]*other.coeff[14] +
             self.coeff[13]*other.coeff[9] + self.coeff[29]*other.coeff[0] +
             other.coeff[29]*self.coeff[0] - 2*self.coeff[15]*other.coeff[29] -
             self.coeff[5]*other.coeff[19] + self.coeff[4]*other.coeff[20] -
             self.coeff[3]*other.coeff[21] - 2*self.coeff[21]*other.coeff[25] +
             self.coeff[1]*other.coeff[25] - self.coeff[20]*other.coeff[4] +
             self.coeff[19]*other.coeff[5] + self.coeff[15]*other.coeff[7] +
             self.coeff[9]*other.coeff[13] - self.coeff[14]*other.coeff[8] +
             self.coeff[7]*other.coeff[15] + 2*self.coeff[25]*other.coeff[21] -
             self.coeff[25]*other.coeff[1] + self.coeff[21]*other.coeff[3],
 -2*self.coeff[30]*other.coeff[15] + self.coeff[12]*other.coeff[13] +
             self.coeff[13]*other.coeff[12] + self.coeff[30]*other.coeff[0] +
             other.coeff[30]*self.coeff[0] - 2*self.coeff[15]*other.coeff[30] -
             self.coeff[11]*other.coeff[14] - self.coeff[5]*other.coeff[22] +
             self.coeff[4]*other.coeff[23] - 2*self.coeff[24]*other.coeff[25] +
             self.coeff[24]*other.coeff[3] - self.coeff[3]*other.coeff[24] +
             self.coeff[2]*other.coeff[25] + self.coeff[15]*other.coeff[10] -
             self.coeff[14]*other.coeff[11] + self.coeff[10]*other.coeff[15] +
             self.coeff[22]*other.coeff[5] + 2*self.coeff[25]*other.coeff[24] -
             self.coeff[25]*other.coeff[2] - self.coeff[23]*other.coeff[4],
 other.coeff[31]*self.coeff[0] + self.coeff[6]*other.coeff[25] +
             2*self.coeff[24]*other.coeff[29] + self.coeff[5]*other.coeff[26] +
             self.coeff[13]*other.coeff[18] + self.coeff[12]*other.coeff[19] -
             self.coeff[11]*other.coeff[20] - self.coeff[4]*other.coeff[27] -
             self.coeff[24]*other.coeff[7] - self.coeff[7]*other.coeff[24] +
             self.coeff[3]*other.coeff[28] - 2*self.coeff[31]*other.coeff[15] -
             2*self.coeff[21]*other.coeff[30] - self.coeff[2]*other.coeff[29] +
             self.coeff[1]*other.coeff[30] - self.coeff[20]*other.coeff[11] +
             self.coeff[19]*other.coeff[12] - 2*self.coeff[30]*other.coeff[21] +
             self.coeff[30]*other.coeff[1] + self.coeff[18]*other.coeff[13] +
             2*self.coeff[29]*other.coeff[24] - self.coeff[29]*other.coeff[2] -
             self.coeff[17]*other.coeff[14] + self.coeff[28]*other.coeff[3] -
             2*self.coeff[28]*other.coeff[25] + self.coeff[15]*other.coeff[16] -
             2*self.coeff[15]*other.coeff[31] - self.coeff[9]*other.coeff[22] +
             self.coeff[8]*other.coeff[23] - self.coeff[14]*other.coeff[17] +
             self.coeff[10]*other.coeff[21] + self.coeff[16]*other.coeff[15] -
             2*self.coeff[25]*other.coeff[28] - self.coeff[22]*other.coeff[9] +
             self.coeff[25]*other.coeff[6] + self.coeff[23]*other.coeff[8] +
             self.coeff[26]*other.coeff[5] - self.coeff[27]*other.coeff[4] +
             self.coeff[21]*other.coeff[10] + self.coeff[31]*other.coeff[0]
 ]
        return cga_object(out)

    def __or__(self, other):
        """inner product

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        out =[
self.coeff[13]*other.coeff[14] + self.coeff[11]*other.coeff[12] +
            self.coeff[23]*other.coeff[22] + self.coeff[12]*other.coeff[11] +
            self.coeff[15]*other.coeff[15] + self.coeff[22]*other.coeff[23] -
            self.coeff[27]*other.coeff[26] + self.coeff[20]*other.coeff[19] -
            self.coeff[5]*other.coeff[4] - self.coeff[10]*other.coeff[10] -
            self.coeff[16]*other.coeff[16] + self.coeff[3]*other.coeff[3] +
            self.coeff[9]*other.coeff[8] - self.coeff[26]*other.coeff[27] +
            self.coeff[14]*other.coeff[13] - self.coeff[7]*other.coeff[7] +
            self.coeff[17]*other.coeff[18] + self.coeff[19]*other.coeff[20] +
            self.coeff[1]*other.coeff[1] + self.coeff[2]*other.coeff[2] +
            self.coeff[8]*other.coeff[9] - self.coeff[4]*other.coeff[5] -
            self.coeff[6]*other.coeff[6] + self.coeff[18]*other.coeff[17],
 -self.coeff[24]*other.coeff[28] + self.coeff[30]*other.coeff[16] +
            self.coeff[11]*other.coeff[18] + self.coeff[4]*other.coeff[9] -
            self.coeff[9]*other.coeff[4] - self.coeff[6]*other.coeff[24] +
            self.coeff[6]*other.coeff[2] + self.coeff[18]*other.coeff[11] -
            self.coeff[7]*other.coeff[25] + self.coeff[21]*other.coeff[15] +
            self.coeff[7]*other.coeff[3] - self.coeff[10]*other.coeff[16] +
            self.coeff[16]*other.coeff[30] - self.coeff[16]*other.coeff[10] +
            self.coeff[19]*other.coeff[14] + self.coeff[20]*other.coeff[13] +
            self.coeff[26]*other.coeff[23] + self.coeff[27]*other.coeff[22] -
            self.coeff[22]*other.coeff[27] - self.coeff[23]*other.coeff[26] -
            self.coeff[3]*other.coeff[7] - self.coeff[2]*other.coeff[6] +
            self.coeff[17]*other.coeff[12] + self.coeff[15]*other.coeff[21] -
            self.coeff[8]*other.coeff[5] + self.coeff[14]*other.coeff[19] +
            self.coeff[13]*other.coeff[20] + self.coeff[5]*other.coeff[8] +
            self.coeff[12]*other.coeff[17] + self.coeff[25]*other.coeff[7] +
            self.coeff[24]*other.coeff[6] - self.coeff[25]*other.coeff[29] -
            self.coeff[31]*other.coeff[30] - self.coeff[30]*other.coeff[31] +
            self.coeff[29]*other.coeff[25] + self.coeff[28]*other.coeff[24],
-self.coeff[29]*other.coeff[16] + self.coeff[14]*other.coeff[22] +
            self.coeff[13]*other.coeff[23] - self.coeff[8]*other.coeff[18] +
            self.coeff[5]*other.coeff[11] - self.coeff[12]*other.coeff[4] -
            self.coeff[11]*other.coeff[5] + self.coeff[25]*other.coeff[10] +
            self.coeff[4]*other.coeff[12] - self.coeff[9]*other.coeff[17] +
            self.coeff[6]*other.coeff[21] - self.coeff[6]*other.coeff[1] -
            self.coeff[18]*other.coeff[8] + self.coeff[7]*other.coeff[16] -
            self.coeff[10]*other.coeff[25] + self.coeff[10]*other.coeff[3] -
            self.coeff[16]*other.coeff[29] + self.coeff[16]*other.coeff[7] -
            self.coeff[21]*other.coeff[6] + self.coeff[24]*other.coeff[15] +
            self.coeff[19]*other.coeff[27] + self.coeff[20]*other.coeff[26] -
            self.coeff[26]*other.coeff[20] - self.coeff[27]*other.coeff[19] +
            self.coeff[22]*other.coeff[14] + self.coeff[23]*other.coeff[13] -
            self.coeff[3]*other.coeff[10] - self.coeff[17]*other.coeff[9] +
            self.coeff[1]*other.coeff[6] + self.coeff[15]*other.coeff[24] -
            self.coeff[25]*other.coeff[30] + self.coeff[21]*other.coeff[28] +
            self.coeff[31]*other.coeff[29] + self.coeff[30]*other.coeff[25] +
            self.coeff[29]*other.coeff[31] - self.coeff[28]*other.coeff[21],
self.coeff[24]*other.coeff[30] - self.coeff[18]*other.coeff[26] -
            self.coeff[6]*other.coeff[16] + self.coeff[7]*other.coeff[21] -
            self.coeff[7]*other.coeff[1] + self.coeff[10]*other.coeff[24] -
            self.coeff[10]*other.coeff[2] + self.coeff[16]*other.coeff[28] -
            self.coeff[16]*other.coeff[6] - self.coeff[21]*other.coeff[7] -
            self.coeff[19]*other.coeff[9] - self.coeff[24]*other.coeff[10] +
            self.coeff[25]*other.coeff[15] - self.coeff[20]*other.coeff[8] +
            self.coeff[26]*other.coeff[18] + self.coeff[27]*other.coeff[17] -
            self.coeff[22]*other.coeff[12] - self.coeff[23]*other.coeff[11] +
            self.coeff[2]*other.coeff[10] + self.coeff[1]*other.coeff[7] -
            self.coeff[17]*other.coeff[27] + self.coeff[15]*other.coeff[25] -
            self.coeff[14]*other.coeff[4] - self.coeff[13]*other.coeff[5] -
            self.coeff[8]*other.coeff[20] + self.coeff[5]*other.coeff[13] -
            self.coeff[12]*other.coeff[22] - self.coeff[11]*other.coeff[23] +
            self.coeff[28]*other.coeff[16] + self.coeff[4]*other.coeff[14] -
            self.coeff[9]*other.coeff[19] + self.coeff[21]*other.coeff[29] -
            self.coeff[31]*other.coeff[28] - self.coeff[30]*other.coeff[24] -
            self.coeff[29]*other.coeff[21] - self.coeff[28]*other.coeff[31],
-2*self.coeff[24]*other.coeff[11] + self.coeff[26]*other.coeff[16] -
            2*self.coeff[25]*other.coeff[13] - self.coeff[19]*other.coeff[7] -
            self.coeff[16]*other.coeff[26] - self.coeff[10]*other.coeff[22] -
            self.coeff[7]*other.coeff[19] - 2*self.coeff[21]*other.coeff[8] -
            self.coeff[22]*other.coeff[10] + self.coeff[3]*other.coeff[13] +
            2*self.coeff[31]*other.coeff[26] - self.coeff[6]*other.coeff[17] +
            self.coeff[2]*other.coeff[11] - self.coeff[17]*other.coeff[6] +
            self.coeff[1]*other.coeff[8] + 2*self.coeff[30]*other.coeff[22] -
            self.coeff[15]*other.coeff[4] - self.coeff[8]*other.coeff[1] -
            self.coeff[13]*other.coeff[3] + 2*self.coeff[29]*other.coeff[19] +
            2*self.coeff[28]*other.coeff[17] - self.coeff[11]*other.coeff[2] +
            self.coeff[4]*other.coeff[15],
2*self.coeff[23]*other.coeff[30] - 2*self.coeff[27]*other.coeff[31] -
            self.coeff[20]*other.coeff[7] + 2*self.coeff[20]*other.coeff[29] -
            self.coeff[16]*other.coeff[27] - self.coeff[18]*other.coeff[6] -
            self.coeff[10]*other.coeff[23] - self.coeff[7]*other.coeff[20] +
            self.coeff[3]*other.coeff[14] - self.coeff[6]*other.coeff[18] +
            self.coeff[2]*other.coeff[12] + 2*self.coeff[18]*other.coeff[28] +
            self.coeff[1]*other.coeff[9] + self.coeff[15]*other.coeff[5] -
            self.coeff[14]*other.coeff[3] - self.coeff[9]*other.coeff[1] +
            2*self.coeff[14]*other.coeff[25] - self.coeff[12]*other.coeff[2] -
            self.coeff[5]*other.coeff[15] + 2*self.coeff[12]*other.coeff[24] -
            self.coeff[23]*other.coeff[10] + 2*self.coeff[9]*other.coeff[21] +
            self.coeff[27]*other.coeff[16],
self.coeff[3]*other.coeff[16] - self.coeff[4]*other.coeff[18] -
            self.coeff[5]*other.coeff[17] + self.coeff[13]*other.coeff[27] +
            self.coeff[14]*other.coeff[26] + self.coeff[15]*other.coeff[28] +
            self.coeff[16]*other.coeff[3] - self.coeff[16]*other.coeff[25] -
            self.coeff[17]*other.coeff[5] - self.coeff[18]*other.coeff[4] -
            self.coeff[25]*other.coeff[16] + self.coeff[25]*other.coeff[31] +
            self.coeff[26]*other.coeff[14] + self.coeff[27]*other.coeff[13] +
            self.coeff[28]*other.coeff[15] + self.coeff[31]*other.coeff[25],
-self.coeff[2]*other.coeff[16] - self.coeff[4]*other.coeff[20] -
            self.coeff[5]*other.coeff[19] - self.coeff[11]*other.coeff[27] -
            self.coeff[12]*other.coeff[26] + self.coeff[15]*other.coeff[29] -
            self.coeff[16]*other.coeff[2] + self.coeff[16]*other.coeff[24] -
            self.coeff[19]*other.coeff[5] - self.coeff[20]*other.coeff[4] +
            self.coeff[24]*other.coeff[16] - self.coeff[24]*other.coeff[31] -
            self.coeff[26]*other.coeff[12] - self.coeff[27]*other.coeff[11] +
            self.coeff[29]*other.coeff[15] - self.coeff[31]*other.coeff[24],
self.coeff[22]*other.coeff[31] - self.coeff[26]*other.coeff[10] +
            self.coeff[26]*other.coeff[30] + self.coeff[25]*other.coeff[19] -
            self.coeff[19]*other.coeff[3] + self.coeff[19]*other.coeff[25] -
            self.coeff[10]*other.coeff[26] - self.coeff[21]*other.coeff[4] +
            self.coeff[24]*other.coeff[17] - self.coeff[3]*other.coeff[19] -
            self.coeff[17]*other.coeff[2] + self.coeff[31]*other.coeff[22] +
            self.coeff[30]*other.coeff[26] - self.coeff[2]*other.coeff[17] +
            self.coeff[17]*other.coeff[24] - self.coeff[13]*other.coeff[29] -
            self.coeff[29]*other.coeff[13] - self.coeff[28]*other.coeff[11] -
            self.coeff[4]*other.coeff[21] - self.coeff[11]*other.coeff[28],
self.coeff[27]*other.coeff[30] + self.coeff[25]*other.coeff[20] -
            self.coeff[20]*other.coeff[3] + self.coeff[20]*other.coeff[25] -
            self.coeff[10]*other.coeff[27] - self.coeff[18]*other.coeff[2] +
            self.coeff[18]*other.coeff[24] + self.coeff[21]*other.coeff[5] -
            self.coeff[23]*other.coeff[31] + self.coeff[24]*other.coeff[18] -
            self.coeff[3]*other.coeff[20] - self.coeff[31]*other.coeff[23] +
            self.coeff[30]*other.coeff[27] - self.coeff[2]*other.coeff[18] +
            self.coeff[14]*other.coeff[29] + self.coeff[29]*other.coeff[14] +
            self.coeff[28]*other.coeff[12] + self.coeff[5]*other.coeff[21] +
            self.coeff[12]*other.coeff[28] - self.coeff[27]*other.coeff[10],
self.coeff[1]*other.coeff[16] - self.coeff[4]*other.coeff[23] -
            self.coeff[5]*other.coeff[22] + self.coeff[8]*other.coeff[27] +
            self.coeff[9]*other.coeff[26] + self.coeff[15]*other.coeff[30] +
            self.coeff[16]*other.coeff[1] - self.coeff[16]*other.coeff[21] -
            self.coeff[21]*other.coeff[16] + self.coeff[21]*other.coeff[31] -
            self.coeff[22]*other.coeff[5] - self.coeff[23]*other.coeff[4] +
            self.coeff[26]*other.coeff[9] + self.coeff[27]*other.coeff[8] +
            self.coeff[30]*other.coeff[15] + self.coeff[31]*other.coeff[21],
-self.coeff[24]*other.coeff[4] + self.coeff[22]*other.coeff[25] +
            self.coeff[26]*other.coeff[7] - self.coeff[26]*other.coeff[29] +
            self.coeff[25]*other.coeff[22] - self.coeff[19]*other.coeff[31] -
            self.coeff[21]*other.coeff[17] + self.coeff[7]*other.coeff[26] -
            self.coeff[22]*other.coeff[3] - self.coeff[3]*other.coeff[22] +
            self.coeff[17]*other.coeff[1] - self.coeff[31]*other.coeff[19] -
            self.coeff[17]*other.coeff[21] + self.coeff[1]*other.coeff[17] -
            self.coeff[30]*other.coeff[13] - self.coeff[13]*other.coeff[30] -
            self.coeff[29]*other.coeff[26] + self.coeff[8]*other.coeff[28] +
            self.coeff[28]*other.coeff[8] - self.coeff[4]*other.coeff[24],
self.coeff[24]*other.coeff[5] - self.coeff[27]*other.coeff[29] +
            self.coeff[25]*other.coeff[23] + self.coeff[20]*other.coeff[31] +
            self.coeff[18]*other.coeff[1] - self.coeff[21]*other.coeff[18] -
            self.coeff[18]*other.coeff[21] + self.coeff[7]*other.coeff[27] -
            self.coeff[3]*other.coeff[23] + self.coeff[31]*other.coeff[20] +
            self.coeff[1]*other.coeff[18] + self.coeff[30]*other.coeff[14] -
            self.coeff[29]*other.coeff[27] + self.coeff[14]*other.coeff[30] -
            self.coeff[9]*other.coeff[28] + self.coeff[5]*other.coeff[24] -
            self.coeff[28]*other.coeff[9] - self.coeff[23]*other.coeff[3] +
            self.coeff[27]*other.coeff[7] + self.coeff[23]*other.coeff[25],
-self.coeff[22]*other.coeff[24] - self.coeff[26]*other.coeff[6] +
            self.coeff[26]*other.coeff[28] - self.coeff[25]*other.coeff[4] +
            self.coeff[19]*other.coeff[1] - self.coeff[19]*other.coeff[21] -
            self.coeff[24]*other.coeff[22] - self.coeff[21]*other.coeff[19] +
            self.coeff[22]*other.coeff[2] - self.coeff[6]*other.coeff[26] +
            self.coeff[31]*other.coeff[17] + self.coeff[17]*other.coeff[31] +
            self.coeff[2]*other.coeff[22] + self.coeff[1]*other.coeff[19] +
            self.coeff[30]*other.coeff[11] + self.coeff[8]*other.coeff[29] +
            self.coeff[29]*other.coeff[8] + self.coeff[28]*other.coeff[26] -
            self.coeff[4]*other.coeff[25] + self.coeff[11]*other.coeff[30],
self.coeff[27]*other.coeff[28] + self.coeff[20]*other.coeff[1] +
            self.coeff[25]*other.coeff[5] - self.coeff[20]*other.coeff[21] -
            self.coeff[24]*other.coeff[23] - self.coeff[21]*other.coeff[20] -
            self.coeff[6]*other.coeff[27] - self.coeff[18]*other.coeff[31] +
            self.coeff[2]*other.coeff[23] - self.coeff[31]*other.coeff[18] +
            self.coeff[1]*other.coeff[20] - self.coeff[30]*other.coeff[12] -
            self.coeff[29]*other.coeff[9] - self.coeff[9]*other.coeff[29] +
            self.coeff[28]*other.coeff[27] + self.coeff[5]*other.coeff[25] -
            self.coeff[12]*other.coeff[30] + self.coeff[23]*other.coeff[2] -
            self.coeff[23]*other.coeff[24] - self.coeff[27]*other.coeff[6],
self.coeff[1]*other.coeff[21] + self.coeff[2]*other.coeff[24] +
            self.coeff[3]*other.coeff[25] - self.coeff[6]*other.coeff[28] -
            self.coeff[7]*other.coeff[29] - self.coeff[10]*other.coeff[30] -
            self.coeff[16]*other.coeff[31] + self.coeff[21]*other.coeff[1] -
            2*self.coeff[21]*other.coeff[21] + self.coeff[24]*other.coeff[2] -
            2*self.coeff[24]*other.coeff[24] + self.coeff[25]*other.coeff[3] -
            2*self.coeff[25]*other.coeff[25] - self.coeff[28]*other.coeff[6] +
            2*self.coeff[28]*other.coeff[28] - self.coeff[29]*other.coeff[7] +
            2*self.coeff[29]*other.coeff[29] - self.coeff[30]*other.coeff[10] +
            2*self.coeff[30]*other.coeff[30] - self.coeff[31]*other.coeff[16] +
            2*self.coeff[31]*other.coeff[31],
self.coeff[4]*other.coeff[27] + self.coeff[5]*other.coeff[26] +
            self.coeff[15]*other.coeff[31] - self.coeff[26]*other.coeff[5] -
            self.coeff[27]*other.coeff[4] + self.coeff[31]*other.coeff[15],
self.coeff[3]*other.coeff[26] + self.coeff[4]*other.coeff[28] -
            self.coeff[13]*other.coeff[31] - self.coeff[25]*other.coeff[26] -
            self.coeff[26]*other.coeff[3] + self.coeff[26]*other.coeff[25] -
            self.coeff[28]*other.coeff[4] - self.coeff[31]*other.coeff[13],
self.coeff[3]*other.coeff[27] - self.coeff[5]*other.coeff[28] +
            self.coeff[14]*other.coeff[31] - self.coeff[25]*other.coeff[27] -
            self.coeff[27]*other.coeff[3] + self.coeff[27]*other.coeff[25] +
            self.coeff[28]*other.coeff[5] + self.coeff[31]*other.coeff[14],
-self.coeff[2]*other.coeff[26] + self.coeff[4]*other.coeff[29] +
            self.coeff[11]*other.coeff[31] + self.coeff[24]*other.coeff[26] +
            self.coeff[26]*other.coeff[2] - self.coeff[26]*other.coeff[24] -
            self.coeff[29]*other.coeff[4] + self.coeff[31]*other.coeff[11],
-self.coeff[2]*other.coeff[27] - self.coeff[5]*other.coeff[29] -
            self.coeff[12]*other.coeff[31] + self.coeff[24]*other.coeff[27] +
            self.coeff[27]*other.coeff[2] - self.coeff[27]*other.coeff[24] +
            self.coeff[29]*other.coeff[5] - self.coeff[31]*other.coeff[12],
-self.coeff[2]*other.coeff[28] - self.coeff[3]*other.coeff[29] -
            self.coeff[10]*other.coeff[31] + self.coeff[24]*other.coeff[28] +
            self.coeff[25]*other.coeff[29] + self.coeff[28]*other.coeff[2] -
            self.coeff[28]*other.coeff[24] + self.coeff[29]*other.coeff[3] -
            self.coeff[29]*other.coeff[25] + self.coeff[30]*other.coeff[31] -
            self.coeff[31]*other.coeff[10] + self.coeff[31]*other.coeff[30],
self.coeff[1]*other.coeff[26] + self.coeff[4]*other.coeff[30] -
            self.coeff[8]*other.coeff[31] - self.coeff[21]*other.coeff[26] -
            self.coeff[26]*other.coeff[1] + self.coeff[26]*other.coeff[21] -
            self.coeff[30]*other.coeff[4] - self.coeff[31]*other.coeff[8],
self.coeff[1]*other.coeff[27] - self.coeff[5]*other.coeff[30] +
            self.coeff[9]*other.coeff[31] - self.coeff[21]*other.coeff[27] -
            self.coeff[27]*other.coeff[1] + self.coeff[27]*other.coeff[21] +
            self.coeff[30]*other.coeff[5] + self.coeff[31]*other.coeff[9],
self.coeff[1]*other.coeff[28] - self.coeff[3]*other.coeff[30] +
            self.coeff[7]*other.coeff[31] - self.coeff[21]*other.coeff[28] +
            self.coeff[25]*other.coeff[30] - self.coeff[28]*other.coeff[1] +
            self.coeff[28]*other.coeff[21] - self.coeff[29]*other.coeff[31] +
            self.coeff[30]*other.coeff[3] - self.coeff[30]*other.coeff[25] +
            self.coeff[31]*other.coeff[7] - self.coeff[31]*other.coeff[29],
self.coeff[1]*other.coeff[29] + self.coeff[2]*other.coeff[30] -
            self.coeff[6]*other.coeff[31] - self.coeff[21]*other.coeff[29] -
            self.coeff[24]*other.coeff[30] + self.coeff[28]*other.coeff[31] -
            self.coeff[29]*other.coeff[1] + self.coeff[29]*other.coeff[21] -
            self.coeff[30]*other.coeff[2] + self.coeff[30]*other.coeff[24] -
            self.coeff[31]*other.coeff[6] + self.coeff[31]*other.coeff[28],
-self.coeff[4]*other.coeff[31] - self.coeff[31]*other.coeff[4],
self.coeff[5]*other.coeff[31] + self.coeff[31]*other.coeff[5],
self.coeff[3]*other.coeff[31] - self.coeff[25]*other.coeff[31] +
            self.coeff[31]*other.coeff[3] - self.coeff[31]*other.coeff[25],
-self.coeff[2]*other.coeff[31] + self.coeff[24]*other.coeff[31] -
            self.coeff[31]*other.coeff[2] + self.coeff[31]*other.coeff[24],
self.coeff[1]*other.coeff[31] - self.coeff[21]*other.coeff[31] +
            self.coeff[31]*other.coeff[1] - self.coeff[31]*other.coeff[21],
0
        ]
        return cga_object(out)

    def __pos__(self):
        """TODO: Docstring for __NEG__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return cga_object(self.coeff)

    def __neg__(self):
        """TODO: Docstring for __NEG__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return cga_object(-self.coeff)

    def __invert__(self):
        """Generates inverse of cga_object"""
        out = [self.coeff[0] - 2*self.coeff[15], self.coeff[1] -
               2*self.coeff[21], self.coeff[2] - 2*self.coeff[24],
               self.coeff[3] - 2*self.coeff[25], self.coeff[4], self.coeff[5],
               -self.coeff[6] + 2*self.coeff[28], -self.coeff[7] +
               2*self.coeff[29], -self.coeff[8], -self.coeff[9],
               -self.coeff[10] + 2*self.coeff[30], -self.coeff[11],
               -self.coeff[12], -self.coeff[13], -self.coeff[14],
               -self.coeff[15], -self.coeff[16] + 2*self.coeff[31],
               -self.coeff[17], -self.coeff[18], -self.coeff[19],
               -self.coeff[20], -self.coeff[21], -self.coeff[22],
               -self.coeff[23], -self.coeff[24], -self.coeff[25],
               self.coeff[26], self.coeff[27], self.coeff[28], self.coeff[29],
               self.coeff[30], self.coeff[31]]
        return cga_object(out)

    def __eq__(self,other):
        """equality checking

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return not(any(self.coeff!=other.coeff))

    def __str__(self):
        """ """
        out = ""
        is_first = True
        for i in range(self.dim):
            if self.coeff[i] != 0:
                if is_first:
                    out += str(self.coeff[i])+self.coeff_names[i]
                    is_first = False
                else:
                    out += " + "+str(self.coeff[i])+"*"+self.coeff_names[i]
        if out == "":
            return "0"
        return out

    def __repr__(self):
        """ """
        return str(self)

    def make_even(self):
        """generates cga_object of even grade with coefficients of self

        Returns: (cga_object) even graded version of self

        Parameters
        ----------

        Returns
        -------


        """
        return cga_object(self.coeff[self.even_indices], even = True)

    def get_even(self):
        """Returns even coefficients of object

        Returns: nd.array

        Parameters
        ----------

        Returns
        -------


        """
        return self.coeff[self.even_indices]


