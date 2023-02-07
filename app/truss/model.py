import json
import re
from io import BytesIO
from typing import List
from typing import Union

import numpy as np
from munch import Munch

from viktor.core import UserException
from viktor.external.generic import GenericAnalysis
from viktor.geometry import Color
from viktor.geometry import Line
from viktor.geometry import Material
from viktor.geometry import Point
from viktor.geometry import RectangularExtrusion
from viktor.utils import memoize
from viktor.views import Label

from .constants import material_allowed_stress
from .constants import profile_properties


class Node(Point):
    """Representing a VIKTOR point but with a tag so we have the same attributes as a RFEM node."""

    def __init__(self, tag: int, x: float, y: float, z: float, **kwargs):
        """Creates a Node with all the attributes a RFEM node has

        Args:
            tag (int): The tag of the node as used in RFEM
            x (float): The x-coordinate that is both used in RFEM and VIKTOR
            y (float): The y-coordinate that is both used in RFEM and VIKTOR
            z (float): The z-coordinate that is both used in RFEM and VIKTOR
        """
        self.tag = tag
        super().__init__(x, y, z)

    def __str__(self):
        return f"{self.tag}:({self.x},{self.y},{self.z})"


class Member(Line):
    """Representing a VIKTOR line but with a tag and bonus attributes so we have the same attributes as a RFEM member."""

    def __init__(
        self,
        nodes: List[Node],
        tag: int,
        start_node: int,
        end_node: int,
        start_section: int = 1,
        end_section: int = 1,
        force: dict = None,
        **kwargs,
    ):
        """Initialises the Member with all the attributes that a RFEM member should have

        Args:
            nodes (List[Node]): The list of nodes already presented so we can connect known nodes.
            tag (int): The tag of the member as used in RFEM
            start_node (int): The tag of the start_node, will be changed to the actual node
            end_node (int): The tag of the end_node, will be changed to the actual node
            start_section (int, optional): The section this member starts in. Defaults to 1.
            end_section (int, optional): The section this member ends in. Defaults to 1.
            force (dict, optional): A dictionary of all the forces on this member. Defaults to None.
        """
        self.tag = tag
        for node in nodes:
            if node.tag == start_node:
                self.start_node = node
            if node.tag == end_node:
                self.end_node = node
        self.force = force
        self.start_section = start_section
        self.end_section = end_section
        super().__init__(self.start_node, self.end_node)

    def __str__(self):
        return f"{self.tag}:({self.start_node},{self.end_node})"


def unity_check(force: float, section_name: str, allowed_stress: float, rounded_val=3) -> float:
    """Returns the unity check factor from the results.

    Args:
        force (float): The force in kN on the member.
        section_name (str): The section_name for the member (e.g. 'SHS 50x50x3.2')
        allowed_stress (float): The allowed stress on the member in Pa
    """
    force *= 1000  # N
    area = profile_properties[section_name]["area"]  # m^2
    calculated_stress = force / area
    return round(float(np.abs(calculated_stress / allowed_stress)), rounded_val)

@memoize
def send2rfem(model: str, selection: int = 0) -> str:
    """Sends a json of the model to the worker. The worker will then build it in rfem
    for solving.

    Args:
        selection (int): The job for the worker. 0->analyse single model, 1->optimize the model

    Returns:
        str: A json with the results of the calculation
    """

    # generic integration
    args = model.encode("utf-8")
    args = BytesIO(args)

    # Using the worker
    files = [("input.json", args)]
    if selection == 0:
        generic_analysis = GenericAnalysis(files=files, executable_key="run_RFEM", output_filenames=["output.json"])
    elif selection == 1:
        generic_analysis = GenericAnalysis(files=files, executable_key="run_OPT", output_filenames=["output.json"])
    else:  # selection not in [0, 1]:
        raise ValueError
    try:
        generic_analysis.execute(timeout=600)
    except TimeoutError as err:
        raise UserException(err)
    output_file = generic_analysis.get_output_file("output.json")
    json_str = output_file.read()
    json_str = json_str.decode("utf-8")
    return json_str


def create_label(member: Member, key: str, decimals: int = 2) -> Label:
    """Create a label with a certain key on the position of the member.

    Args:
        member (Member): The member you want to create a label for.
        key (str): The key for the force dictionary from member.
        decimals (int, optional): The number of decimal to display. Defaults to 2.
    """

    def avg(v1, v2):
        """Average of two points"""
        return float((v2 + v1) / 2)

    sp = member.start_point
    ep = member.end_point
    x = avg(sp.x, ep.x)
    y = avg(sp.y, ep.y)
    z = avg(sp.z, ep.z)

    text = str(round(member.force[key], decimals))  # Member has different forces, print the one selected by the key
    return Label(Point(x, y, z), text)


def rgb(minimum: float, maximum: float, value: float) -> Union[int, int, int]:
    """
    Creates the rgb values to visualise the forces. Code found on stackoverflow:
    https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
    """
    try:
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
    except ZeroDivisionError:
        # Some forces do not apply so we get a division by zero error
        r, g, b = 221, 221, 221
    return r, g, b


class Model:
    """The Model class is the bridge between VIKTOR and RFEM. It can both be visualized inside VIKTOR but can also be exported to RFEM."""

    def __init__(self, model_dict: dict = None, safety_factor: float = None):
        """It is possible to initialise the model from a dict file. This dict file is created by the build_rfem_model script."""
        self.materials = []
        self.sections = []
        self.nodes = []
        self.members = []
        self.simulation = dict()

        if model_dict is not None:
            self.from_dict(model_dict)
        if safety_factor is not None:
            self.compute_unity_check_by_member(safety_factor)

    def visualize(self, params: Munch = None):
        """Create RectangularExtrusions for each member in the model"""
        self.labels = []
        geometries = []
        for member in self.members:
            for section in self.sections:
                # Get the right material for each member
                if member.start_section == section["tag"]:
                    # Parsing the string input to dimensions we can use
                    dimensions = re.search(r"[0-9]+x[0-9]+x[0-9]+", section["material"])[0].split("x")
                    break

            # Only add colors if there is some force on the member
            if member.force is None:  # Used for design
                material = None
            else:  # Used for analysis
                force_selection = params.step_analyse.force
                force = member.force[force_selection]
                r, g, b = rgb(self.min_force(key=force_selection), self.max_force(key=force_selection), force)
                if params.step_analyse.show_labels:
                    opacity = 0.5
                else:
                    opacity = 1
                material = Material(str(force), color=Color(r, g, b), threejs_opacity=opacity)
                decimals = params.step_analyse.decimals
                self.labels.append(create_label(member, force_selection, decimals))

            geometries.append(
                RectangularExtrusion(
                    float(dimensions[0]) * 0.001, float(dimensions[1]) * 0.001, member, material=material
                )
            )
        return geometries

    def to_dict(self) -> dict:
        """Export the model to a dictionary"""
        output = {"materials": [], "sections": [], "nodes": [], "members": []}
        for material in self.materials:
            output["materials"].append(material)
        for section in self.sections:
            output["sections"].append(section)
        for node in self.nodes:
            output["nodes"].append({"tag": int(node.tag), "x": node.x, "y": node.y, "z": node.z})
        for member in self.members:
            output["members"].append(
                {
                    "tag": int(member.tag),
                    "start_node": int(member.start_node.tag),
                    "end_node": int(member.end_node.tag),
                    "rotation_angle": 0.0,
                    "start_section": int(member.start_section),
                    "end_section": int(member.end_section),
                }
            )
        output["simulation"] = self.simulation
        return output

    def to_json(self) -> str:
        """Export the model to a readable json so we can export it."""
        return json.dumps(self.to_dict(), indent=4)

    def from_dict(self, model_dict) -> None:
        """Import function to create a model"""
        self.simulation = model_dict["simulation"]
        for material in model_dict["materials"]:
            self.materials.append(material)
        for section in model_dict["sections"]:
            self.sections.append(section)
        for node in model_dict["nodes"]:
            self.nodes.append(Node(tag=node["tag"], x=node["x"], y=node["y"], z=node["z"]))
        for member in model_dict["members"]:
            # if not member.get('force'):
            #     member['force'] = {'N': None, 'V_y': None, 'V_z': None, 'M_T': None, 'M_y': None, 'M_z': None}
            self.members.append(
                Member(
                    nodes=self.nodes,
                    tag=member["tag"],
                    start_node=member["start_node"],
                    end_node=member["end_node"],
                    start_section=member["start_section"],
                    end_section=member["end_section"],
                    force=member.get('force'),
                )
            )

    def max_force(self, key):
        """From all the forces on the members for a certain key, get the maximum"""
        x = -np.inf
        for member in self.members:
            if not member.force:
                continue
            y = member.force[key]
            if y > x:
                x = y
        return x

    def min_force(self, key):
        """From all the forces on the members for a certain key, get the minimum"""
        x = np.inf
        for member in self.members:
            if not member.force:
                continue
            y = member.force[key]
            if y < x:
                x = y
        return x

    def compute_unity_check_by_member(self, safety_factor: float) -> None:
        material = self.materials[0]["name"]
        allowed_stress = material_allowed_stress[material]["low"] * 1e6  # Pa = N/m^2
        allowed_stress *= safety_factor
        for member in self.members:
            if not member.force:
                continue
            member.force["UC"] = unity_check(
                member.force["N"], self.sections[member.start_section - 1]["material"], allowed_stress
            )


class Truss(Model):
    """The Truss is a model build using parameters."""

    def __init__(
        self,
        length: float = 10,
        height: float = 6,
        height_panels: float = 1,
        truss_panels: float = 8,
        simulation: dict = None,
        sections: list = [],
        materials: list = [],
    ):
        """Creates a Truss using parameters.

        Args:
            length (float, optional): Length of the truss in m. Defaults to 10.
            height (float, optional): Height of the truss in m. Defaults to 6.
            height_panels (float, optional): Height of the panels in m. Defaults to 1.
            truss_panels (float, optional): Number of truss panels. Defaults to 8.
            simulation (dict, optional): The parameters for the simulation. Defaults to None.
            sections (list, optional): List of sections the truss contains (e.g. 'SHS 50x50x3.2'). Defaults to [].
            materials (list, optional): List of materials the truss contains (e.g. 'S355'). Defaults to [].
        """
        super().__init__()
        self.simulation = simulation

        for idx, material in enumerate(materials):
            self.materials.append({"tag": idx + 1, "name": material})

        for idx, section in enumerate(sections):
            self.sections.append({"tag": idx + 1, "material": section})

        # input control
        if truss_panels % 2 != 0:  # Number of panels should be divisable by 2
            truss_panels += 1

        # Create Nodes
        x_nodes = np.repeat(np.arange(0, length + length / truss_panels - 0.1, length / truss_panels), 2)
        z_nodes = (height + height_panels, height) * int((len(x_nodes) / 2))
        tag_nodes = np.arange(1, len(x_nodes) + 1, 1)

        for tag, x, z in zip(tag_nodes, x_nodes, z_nodes):
            self.nodes.append(Node(tag, x, 0, z))

        # Create Lower Chord
        self.members.append(Member(self.nodes, 1, 1, tag_nodes[-2], 1, 1))

        # Create Upper Chord
        self.members.append(Member(self.nodes, 2, 2, tag_nodes[-1], 1, 1))

        # Create Verticals
        i = 3
        j = 1
        while j < len(tag_nodes) and i < len(tag_nodes) - 1:
            self.members.append(Member(self.nodes, j + 2, i, i + 1, 2, 2))
            i += 2
            j += 1

        # Create Diagonals
        diagonal_tag = np.arange((len(tag_nodes) / 2 + 3), (len(tag_nodes) / 2 + 3) + truss_panels, 1)
        i = 1
        j = int(diagonal_tag[0])
        while i < (tag_nodes[-1] / 2) and j < diagonal_tag[-1] + 1:
            self.members.append(Member(self.nodes, j, i + 1, i + 2, 3, 3))
            i += 2
            j += 1
        i = int(len(tag_nodes) / 2)
        j = int(diagonal_tag[int(len(diagonal_tag) / 2)])
        while i < tag_nodes[-1] and j < diagonal_tag[-1] + 1:
            self.members.append(Member(self.nodes, j, i, i + 3, 3, 3))
            i += 2
            j += 1

        # Create Columns
        column_tags = np.arange(tag_nodes[-1] + 1, tag_nodes[-1] + 3, 1)
        self.nodes.append(Node(column_tags[0], 0.0, 0.0, 0.0))
        self.nodes.append(Node(column_tags[1], length, 0.0, 0.0))
        self.members.append(Member(self.nodes, j + 1, column_tags[0], tag_nodes[0], 4, 4))
        self.members.append(Member(self.nodes, j + 2, column_tags[1], tag_nodes[-2], 4, 4))
