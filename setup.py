from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path: str)->List[str]:
    '''
    This function will return the list of requiremnets
    '''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        print("requirements", requirements)

    return requirements


setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Smita',
    author_email = 'me.smitameshram@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)

