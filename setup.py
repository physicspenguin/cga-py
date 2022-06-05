from setuptools import setup

me = "Daren Thimm"
my_email = "daren.thimm@gmx.at"
setup(
    name="cga-py",
    version="1.0.0",
    packages=[
        "cga_py",
    ],
    scripts=["cga_py/bin/cga_py_vis.py"],
    license="BSD-3-Clause License",
    author=me,
    author_email=my_email,
    maintainer=me,
    maintainer_email=my_email,
    description="Implementation of CGA for python",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
)
