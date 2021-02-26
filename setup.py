from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# Tip: Generate requirements using pigar: pigar -p requirements.txt -P magnet_loss_iclr2016/
setup(
    name="magnet_loss_iclr2016",
    version="0.0.1",
    description="Play a PIL Image list as a video",
    author=" Kodur Krishna Chaitanya",
    author_email="kodur.chaitanya@colorado.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heracleia/Magnet-Loss",
    install_requires=parse_requirements("requirements.txt"),
    packages=find_packages(exclude=("tests", "docs")),
)
