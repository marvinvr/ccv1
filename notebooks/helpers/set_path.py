import os
import sys


def add_project_root_to_sys_path():
    """
    Adds the project root directory to the system path.

    This function gets the absolute path of the parent directory of the current working directory and adds it to the system path if it is not already there.
    """
    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    if project_root not in sys.path:
        sys.path.append(project_root)


add_project_root_to_sys_path()
