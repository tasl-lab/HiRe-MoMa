from setuptools import find_packages, setup

setup(
    name="hire_moma_launcher",
    version="0.1.0",
    description="HiRe-MoMa agents, networks and replay buffers (built on SERL / HIL-SERL)",
    packages=find_packages(include=["hire_moma_launcher", "hire_moma_launcher.*"]),
    python_requires=">=3.10",
    install_requires=[
        "jax==0.4.35",
        "flax==0.10.2",
        "optax==0.2.4",
        "chex==0.1.87",
        "distrax==0.1.5",
        "tensorflow-probability==0.25.0",
        "gymnasium==0.29.1",
        "numpy==1.26.4",
        "scipy",
        "einops",
        "opencv-python",
        "requests",
        "wandb",
        "agentlace @ git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
    ],
)
