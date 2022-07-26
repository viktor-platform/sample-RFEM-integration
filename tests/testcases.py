import unittest
from pathlib import Path
from munch import Munch
import json

from app.truss.controller import build_model_from_params


def mock_params():
    '''Simulate default input of the parametrization'''
    params = Munch()
    params.step_design = Munch()
    params.step_design.frame = Munch()
    params.step_design.sections = Munch()
    params.step_design.simulation = Munch()

    params.step_design.frame.length = 10
    params.step_design.frame.height = 6
    params.step_design.frame.height_panels = 1
    params.step_design.sections.material1 = 'SHS 50x50x4'
    params.step_design.sections.material2 = 'SHS 50x50x4'
    params.step_design.sections.material3 = 'SHS 50x50x4'
    params.step_design.sections.material4 = 'SHS 50x50x4'
    params.step_design.sections.material = 'S355'

    params.step_design.simulation.self_weight = 1

    return params

def mock_model_input():
    with open(Path(__file__).parent / 'lib' / 'input.json') as f:
        return f.read()

class TestRfem(unittest.TestCase):

    def test_to_json(self):
        """Check if the model build using the default parameters builds the default model"""
        params = mock_params()
        model = json.loads(build_model_from_params(params).to_json())
        input_json = json.loads(mock_model_input())

        for (key, value1), (_, value2) in zip(model.items(),input_json.items()):
            with self.subTest(msg=key):
                self.assertEqual(value1,value2)

if __name__ == '__main__':
    unittest.main()