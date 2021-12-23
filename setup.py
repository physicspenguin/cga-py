from distutils.core import setup
me = 'Daren Thimm'
my_email = 'daren.thimm@student.uibk.ac.at'
setup(
    name='CGApy',
    version='0.1dev',
    packages=['CGApy',],
    license='To be determined',
    author=me,
    author_email=my_email,
    maintainer=me,
    maintainer_email=my_email
    description='Implementation of the CGA',
    long_description=open('./README.md').read()
)
