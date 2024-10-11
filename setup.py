from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "google-cloud-storage",
    "transformers[torch]",
    "datasets",
    "tqdm",
    "cloudml-hypertune",
    "scikit-learn>=1.2.2",
    # "gcsfs",
]

setup(
    name="trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Vertex AI | Training | PyTorch | Text Classification | Python Package",
)
