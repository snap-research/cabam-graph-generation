from setuptools import setup, find_packages


def get_req_list_from_req_file():
    with open("./requirements.txt") as f:
        requirements = [req.strip() for req in f.readlines()]
    return requirements


setup(
    name="cabam",
    version="0.1.0",
    packages=find_packages(include=["cabam", "cabam.*"]),
    install_requires=get_req_list_from_req_file(),
)
