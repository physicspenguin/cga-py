from setuptools import setup

me = "Daren Thimm"
my_email = "daren.thimm@gmx.at"
setup(
    name="cga-py",
    version="0.0.1-dev",
    packages=[
        "cga_py",
    ],
    license="MIT",
    author=me,
    author_email=my_email,
    maintainer=me,
    maintainer_email=my_email,
    description="Implementation of CGA for python",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
)
