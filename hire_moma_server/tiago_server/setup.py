from setuptools import find_packages, setup

setup(
    name="tiago_server",
    version="0.1.0",
    description="HiRe-MoMa TIAGo++ environment client, experiments and training scripts",
    packages=find_packages(include=["tiago_server", "tiago_server.*"]),
    python_requires=">=3.10",
    install_requires=["hire_moma_launcher", "absl-py", "tqdm", "natsort", "requests", "pynput"],
)
