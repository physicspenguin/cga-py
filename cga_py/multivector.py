import numpy as np

class cga_object:

    """
    Element of the CGA with methods for:
    addition, multiplication,
    printing

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
        if isinstance(gen,cga_object):
            cof = gen.coeff
        else:
            cof = gen
        ## Version if list is given
        self.coeff = np.zeros(self.dim)
        for i in range(len(cof)):
            self.coeff[i] = cof[i]

    def __add__(self, other):
        """Addition of cga_object

        Args:
            other (cga_object): object to add to self

        Returns: addition of cga_object with cga_object

        """
        # This is a rather hacky way of ensuring that addition of different
        # datatypes is commutative. Pending better implementation
        try:
            coefficients = self.coeff + other.coeff
        except:
            coefficients = (other + self).coeff
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
# Base Objects
########################################

e_1     = cga_object(np.eye(32)[:,1])
e_2     = cga_object(np.eye(32)[:,2])
e_3     = cga_object(np.eye(32)[:,3])
e_i     = cga_object(np.eye(32)[:,4])
e_o     = cga_object(np.eye(32)[:,5])
e_12    = cga_object(np.eye(32)[:,6])
e_13    = cga_object(np.eye(32)[:,7])
e_1i    = cga_object(np.eye(32)[:,8])
e_1o    = cga_object(np.eye(32)[:,9])
e_23    = cga_object(np.eye(32)[:,10])
e_2i    = cga_object(np.eye(32)[:,11])
e_2o    = cga_object(np.eye(32)[:,12])
e_3i    = cga_object(np.eye(32)[:,13])
e_3o    = cga_object(np.eye(32)[:,14])
e_io    = cga_object(np.eye(32)[:,15])
e_123   = cga_object(np.eye(32)[:,16])
e_12i   = cga_object(np.eye(32)[:,17])
e_12o   = cga_object(np.eye(32)[:,18])
e_13i   = cga_object(np.eye(32)[:,19])
e_13o   = cga_object(np.eye(32)[:,20])
e_1io   = cga_object(np.eye(32)[:,21])
e_23i   = cga_object(np.eye(32)[:,22])
e_23o   = cga_object(np.eye(32)[:,23])
e_2io   = cga_object(np.eye(32)[:,24])
e_3io   = cga_object(np.eye(32)[:,25])
e_123i  = cga_object(np.eye(32)[:,26])
e_123o  = cga_object(np.eye(32)[:,27])
e_12io  = cga_object(np.eye(32)[:,28])
e_13io  = cga_object(np.eye(32)[:,29])
e_23io  = cga_object(np.eye(32)[:,30])
e_123io = cga_object(np.eye(32)[:,31])


########################################
# Testing
########################################
d = cga_object([1,1,0,3])

e = cga_object([0,1,1,0])

f = cga_object([1,1,1,0])
