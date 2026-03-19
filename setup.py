from setuptools import setup, find_packages

setup(
    name="congestion_experiments",
    version="0.1.0",
    description="Estimators for experimenting under stochastic congestion (Li, Johari, Kuang, Wager)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy", "scipy"],
)
