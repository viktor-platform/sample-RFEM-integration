from viktor.parametrization import BooleanField
from viktor.parametrization import IsTrue
from viktor.parametrization import Lookup
from viktor.parametrization import NumberField
from viktor.parametrization import OptimizationButton
from viktor.parametrization import OptionField
from viktor.parametrization import Parametrization
from viktor.parametrization import Section
from viktor.parametrization import Step

from .constants import force_options
from .constants import material_options
from .constants import profiles_options


class BuildingParametrization(Parametrization):
    step_design = Step("Design", views=["visualize_truss"])
    step_design.frame = Section("Frame")
    step_design.frame.length = NumberField("Length", default=10, min=1, max=100)
    step_design.frame.height = NumberField("Height", default=6, min=1, max=100)
    step_design.frame.height_panels = NumberField("Truss height", default=1, min=1, max=100, step=0.2)

    step_design.sections = Section("Sections")
    step_design.sections.material1 = OptionField(
        "Chords", options=profiles_options, default="SHS 50x50x4", description="The top and bottom beams."
    )
    step_design.sections.material2 = OptionField("Vertical members", options=profiles_options, default="SHS 50x50x4")
    step_design.sections.material3 = OptionField("Diagonal members", options=profiles_options, default="SHS 50x50x4")
    step_design.sections.material4 = OptionField("Columns", options=profiles_options, default="SHS 50x50x4")

    step_design.loads_materials = Section("Loads and materials")
    step_design.loads_materials.material = OptionField("Material", options=material_options, default="S235")
    step_design.loads_materials.safety_factor = NumberField(
        "Material reduction factor", min=0.1, max=1, default=0.9, step=0.1
    )
    step_design.loads_materials.member_load = NumberField(
        "Distributed load",
        default=100,
        suffix="kN/m",
    )

    step_design.optimization = Section("Optimization")
    step_design.optimization.min = NumberField("Minimum truss height", min=0.1, max=100, default=1, step=0.1)
    step_design.optimization.max = NumberField("Maximum truss height", min=0.1, max=100, default=5, step=0.1)
    step_design.optimization.steps = NumberField(
        "Steps",
        min=1,
        max=10,
        default=3,
        description="Number of optimisation steps between minimum and maximum. Keep in mind that this will increase computation time.",
    )
    step_design.optimization.optimize_height = OptimizationButton(
        "Optimize truss height", "get_optimal_result", longpoll=True
    )

    step_analyse = Step("Analyse", views=["analyse_rfem"])
    step_analyse.force = OptionField("Display", options=force_options, default="N", flex=50)
    step_analyse.show_labels = BooleanField("Show labels", default=True)
    step_analyse.decimals = NumberField(
        "Number of decimals", min=1, default=2, visible=IsTrue(Lookup("step_analyse.show_labels"))
    )
