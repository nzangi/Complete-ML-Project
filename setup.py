from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    The function will return a list of requirements for a given
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(
name = "Complete_ML_Project",
version= "0.0.1",
author="nzangi",
author_email="nzangimuoki284@gmail.com",
packages=find_packages(),
install_requires= get_requirements('requirements.txt')
)