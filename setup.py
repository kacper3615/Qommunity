from setuptools import setup, find_packages

setup(
    name="Qommunity",
    version="0.1.0",
    description="Library for community detection where you can use variety of solvers, including quantum solvers",
    packages=find_packages(),
    python_requires=">=3.9,<=3.11",
    install_requires=[
        "bayanpy==0.7.7",
        "dimod==0.12.16",
        "dwave_cloud_client==0.12.0",
        "dwave_optimization==0.1.0",
        "dwave_preprocessing==0.6.6",
        "dwave_samplers==1.3.0",
        "dwave_system==1.24.0",
        "leidenalg==0.10.2",
        "networkx==3.3",
        "pytest==8.2.2",
        "python_igraph==0.11.6",
        "powerlaw",
        "QHyper==0.3.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)