import json
from io import StringIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from munch import Munch

from app.truss.parametrization import BuildingParametrization
from viktor import ViktorController
from viktor.result import OptimizationResult
from viktor.result import OptimizationResultElement
from viktor.views import GeometryAndDataResult
from viktor.views import GeometryAndDataView
from viktor.views import GeometryResult
from viktor.views import GeometryView
from viktor.views import SVGResult

from .constants import material_allowed_stress
from .datagroups_and_visualisations import build_model_from_params
from .datagroups_and_visualisations import create_datagroup
from .model import Model
from .model import send2rfem
from .model import unity_check


class Controller(ViktorController):
    label = "RFEM"
    parametrization = BuildingParametrization

    @GeometryView("Design", duration_guess=3)
    def visualize_truss(self, params, **kwargs):
        """Build the model and visualize it in a geometry view"""
        model = build_model_from_params(params)
        return GeometryResult(model.visualize())

    @GeometryAndDataView("Data", duration_guess=10)
    def analyse_rfem(self, params, **kwargs):
        """Build the model to be send to RFEM and after calculation renders a new model with the received output"""
        # Communicate with RFEM
        model = build_model_from_params(params)
        json_string = send2rfem(model.to_json(), 0)
        model_dict = json.loads(json_string)

        # Create a datagroup for the DataView
        safety_factor = params.step_design.loads_materials.safety_factor
        data = create_datagroup(model_dict, safety_factor)

        # Input the results calculated by RFEM to the model
        new_model = Model(model_dict, params.step_design.loads_materials.safety_factor)
        geometry = new_model.visualize(params)

        # Toggle to show the labels or not
        if params.step_analyse.show_labels:
            labels = new_model.labels
        else:
            labels = None

        return GeometryAndDataResult(geometry, data, labels=labels)

    def get_optimal_result(self, params: Munch, **kwargs) -> OptimizationResult:
        """Builds multiple models to be evaluated and plot the unity check."""

        # Builds the models
        l_min = params.step_design.optimization.min
        l_max = params.step_design.optimization.max
        steps = params.step_design.optimization.steps
        models = []
        params_used = []
        x = np.linspace(l_min, l_max, steps)
        for length in x:
            # Override the length parameter
            params.step_design.frame.height_panels = length
            params_used.append(params.copy())
            model = build_model_from_params(params)
            models.append(model.to_dict())
        json_string = json.dumps(models, indent=4)

        # Send the models to RFEM
        output = json.loads(send2rfem(json_string, 1))

        # Build optimization result
        allowed_stress = material_allowed_stress[params.step_design.loads_materials.material]["low"] * 1e6  # Pa = N/m^2
        allowed_stress *= params.step_design.loads_materials.safety_factor
        results = []
        results_array = []
        for raw, old_params in zip(output, params_used):
            result = dict()
            result["Chords"] = unity_check(raw["1"]["N"], raw["1"]["material"], allowed_stress)
            result["Vertical members"] = unity_check(raw["2"]["N"], raw["2"]["material"], allowed_stress)
            result["Diagonal members"] = unity_check(raw["3"]["N"], raw["3"]["material"], allowed_stress)
            result["Columns"] = unity_check(raw["4"]["N"], raw["4"]["material"], allowed_stress)
            results.append(OptimizationResultElement(old_params, result))
            results_array.append(result)

        # Plot figure
        io_ = StringIO()
        df = pd.DataFrame(results_array)
        df["Mean"] = df.mean(axis=1)
        df["x"] = x
        ax = plt.gca()
        for y in df.columns[:-2]:
            df.plot(kind="line", x="x", y=y, ax=ax)
        df.plot(kind="line", x="x", y="Mean", ax=ax, style="--")
        ax.set_title("Unity Check per section")
        ax.set_xlabel("Height panels")
        ax.set_ylabel("UC")
        plt.savefig(io_, format="svg")
        image = SVGResult(io_)

        # Return results
        return OptimizationResult(
            results,
            ["step_design.frame.height_panels"],
            output_headers={
                "Chords": "Chords",
                "Vertical members": "Vertical members",
                "Diagonal members": "Diagonal members",
                "Columns": "Columns",
            },
            image=image,
        )
