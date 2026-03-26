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

setup(version=fastpt_version, packages=find_packages())