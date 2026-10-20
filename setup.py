from setuptools import find_packages, setup


def get_requirements():
    requirements = []
    try:
        with open("requirements.txt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line != "-e .":
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements


setup(
    name="frauddetection",
    version="0.1.0",
    description="Détection de fraude bancaire — démo cycle de vie MLflow",
    author="NDA",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.10",
)
