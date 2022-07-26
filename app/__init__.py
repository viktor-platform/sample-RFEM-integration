from viktor import InitialEntity

from .truss.controller import Controller as TrussController

initial_entities = [InitialEntity("TrussController", name="Truss")]
