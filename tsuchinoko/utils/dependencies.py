import importlib
import subprocess
import sys
from typing import List


def oxfordcomma(listed):
    if len(listed) == 0:
        return ''
    if len(listed) == 1:
        return listed[0]
    if len(listed) == 2:
        return listed[0] + ' and ' + listed[1]
    return ', '.join(listed[:-1]) + ', and ' + listed[-1]


def ensure_dependencies(deps: List[str]):
    install_warning = """
    ATTENTION!
    This demo depends on a package that is not included in the Tsuchinoko installation.
    As a convenience, Tsuchinoko may install the required dependencies for you.
    """

    try:
        from perlin_noise import PerlinNoise
    except ImportError:
        print(install_warning)
        response = input(f'Would you like to install {oxfordcomma(deps)} [y/(n)]? ')
        if response != 'y':
            print('Aborting...')
            sys.exit(1)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])


def check_dependencies(deps: List[str]) -> bool:
    for dep in deps:
        if dep in sys.modules:
            print(f"{dep!r} already in sys.modules")
        elif (spec := importlib.util.find_spec(dep)) is not None:
            pass
            # module = importlib.util.module_from_spec(spec)
            # sys.modules[dep] = module
            # spec.loader.exec_module(module)
            # print(f"{dep!r} has been imported")
        else:
            return False
    return True