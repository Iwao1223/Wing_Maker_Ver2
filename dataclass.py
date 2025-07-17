# dataclass.py
from dataclasses import dataclass
import numpy as np


@dataclass
class WingConfig:
    rib_station_list: np.ndarray
    strut_station_list: np.ndarray
    brock_span: int
    chord_R: int
    chord_T: int
    airfoil_R: str
    airfoil_T: str
    spar_major_diameter_R: float
    spar_major_diameter_T: float
    spar_minor_diameter_R: float
    spar_minor_diameter_T: float
    spar_position_R: float
    spar_position_T: float
    ellipse_degree_R : float
    ellipse_degree_T: float
    rear_spar_diameter_R: float
    rear_spar_diameter_T: float
    rear_spar_position_R: float
    rear_spar_position_T: float
    aileron_diameter_coef : float
    aileron_position: float
    plank_thickness: float
    rib_cap_thickness: float
    plank_position_top: float
    plank_position_bottom: float
    rib_cap_position: float
    spar_cap_thickness: float
    spar_cap_degree: float
    squid_cap_thickness: float
    squid_cap_length: float
    squid_cap_text_wide: float
    joint_thickness: float
    squid_length: float
    te_length: float
    te_carbon_thickness: float
    rib_interval: int
    rib_thickness: float
    strut_thickness: float
    x_bracing_pit_diameter: float
    brock_name: str
    brock_number: int
    text_position: float
    left_or_right: int
    joint_width: float
    plank: bool
    rib_cap_leading: bool
    rib_cap_trailing: bool
    spar_cap: bool
    rear_spar_cap: bool
    squid: bool
    only_rib: bool
    only_plank: bool
    circle_or_ellipse: str
