from cga_py import *
from scipy.optimize import root as solve
a = e_12 + 3*e_1o
b = e_1 + e_12
# print(rand_rotor())
# s = sphere([3,2,1], 4)
# print(sphere_to_cartesian(s))
# print(s)
# s = 2*s
# print(s)
# print(normalize_sphere(s))
# print(sphere_to_cartesian(s))
# p = point([1,2,3])
# print(p)
# p = 2*p
# print(p)
# print(normalize_point(p))
# print(point_to_cartesian(p))

# p = rand_point()
# print(p)
# print(point_to_cartesian(p))

# s = rand_sphere()
# print(s)
# print(sphere_to_cartesian(s))

# pl = plane([1,2,3],4)

def rand_rotor(maximum = 10):
    """Generates random rotor with rational coefficients

    Kwargs:
        maximum (float): maximum value

    Returns: TODO

    """
    x = np.array([rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum),
                  rand_rational(maximum)])
    # study = lambda x12,x13,x14,x15,x16: np.linalg.norm(study_var(cga_object(np.append(x,np.array([x12,x13,x14,x15,x16])),True))) * np.ones(5)
    study = lambda y: study_var(cga_object(np.append(x,y),True))[0:5]
    # print(study([rand_rational(),rand_rational(),rand_rational(),rand_rational(),rand_rational()]))
    out = solve(study,np.ones(5), method="krylov")
    print(out)
    # print(study(out))
    return

rand_rotor()

# def testing(x):
    # """TODO: Docstring for testing.

    # Args:
        # x (TODO): TODO

    # Returns: TODO

    # """
    # return np.linalg.norm(2*x**2+3*x)

# solve(testing,[0,1,2,3,4])

