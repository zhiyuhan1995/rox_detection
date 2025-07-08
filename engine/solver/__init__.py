"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .yolo_solver import YoloSolver




from typing import Dict

TASKS :Dict[str, BaseSolver|YoloSolver] = {
    'classification': ClasSolver,
    'detection': DetSolver,
    'yolo': YoloSolver
}
