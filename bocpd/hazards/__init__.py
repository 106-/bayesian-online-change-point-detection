"""ベイズオンライン変化点検知のためのハザード関数"""

from bocpd.hazards.base import HazardFunction
from bocpd.hazards.constant import ConstantHazard

__all__ = ["HazardFunction", "ConstantHazard"]
