from setuptools import setup, find_packages

setup(
    name="snn2",
    version="0.1.0",
    description="Spec-driven, batched, validated spiking-net experiments.",
    packages=find_packages(),
    install_requires=["numpy>=1.21"],
    extras_require={"scale": ["ray[tune]>=2.0"]},
    python_requires=">=3.9",
)
