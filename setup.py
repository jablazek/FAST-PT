import os, re
from setuptools import setup, find_packages

# Read in the fastpt version from fastpt/info.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('fastpt','info.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    fastpt_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('FASTPT version is %s'%(fastpt_version))

setup(
    name='fast-pt',
    description="Fast cosmological perturbation theory",
    long_description=(
        "FAST-PT is a code to calculate quantities in cosmological "
        "perturbation theory at 1-loop (including, e.g., corrections to the "
        "matter power spectrum)."),
    author="Joseph E. McEwen, Xiao Fang, and Jonathan Blazek (blazek@berkeley.edu)",
    author_email="blazek@berkeley.edu",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    version=fastpt_version,
    license="MIT license",
    url="https://github.com/JoeMcEwen/FAST-PT",
    download_url="https://github.com/JoeMcEwen/FAST-PT/archive/v%s.zip"%fastpt_version,
    
)
