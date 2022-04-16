from setuptools import setup

me = "Daren Thimm"
my_email = "daren.thimm@student.uibk.ac.at"
setup(
    name="cga-py",
    version="0.0.1-dev",
    packages=[
        "cga_py",
    ],
    license="Apache 2",
    author=me,
    author_email=my_email,
    maintainer=me,
    maintainer_email=my_email,
    description="Implementation of the CGA",
    long_description=open("./README.md").read(),
)
