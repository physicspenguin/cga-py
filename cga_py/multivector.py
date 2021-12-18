import numpy as np

class cga_object:

    """


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

    def __init__(self, gen):
        ## Version if list is given
        self.coeff = np.zeros(self.dim)
        for i in range(len(gen)):
            self.coeff[i] = gen[i]

    def __add__(self, other):
        """Addition of cga_object

        Args:
            other (cga_object): object to add to self

        Returns: addition of cga_object with cga_object

        """
        coefficients = self.coeff + other.coeff
        return cga_object(coefficients)

    def __radd__(self, other):
        """Addition of cga_object

        Args:
            other (float): object to add to self

        Returns: addition of cga_object with scalar object

        """
        return cga_object((cga_object([other])+self).coeff)

    def __mul__(self, other):
        """CGA multiplication of two cga_object

        Args:
            other (cga_object): cga_object to multiply with self

        Returns: (cga_object) multiplication of other and self

        """
        pass

    def __rmul__(self,other):
        cof = np.zeros(self.dim)
        for i in range(self.dim):
            cof[i] = other*self.coeff[i]
        return cga_object(cof)


    def pprint(self):
        out = ""
        is_first = True
        for i in range(self.dim):
            if self.coeff[i] != 0:
                if is_first:
                    out += str(self.coeff[i])+self.coeff_names[i]
                    is_first = False
                else:
                    out += " + "+str(self.coeff[i])+self.coeff_names[i]
        print(out)






########################################
# Testing
########################################
d = cga_object([1,1,0,3])

e = cga_object([0,1,1,0])

f = cga_object([1,1,1,0])
