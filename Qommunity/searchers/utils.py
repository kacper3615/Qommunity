# from enum import Enum
# from QHyper.solvers.base import SamplesetData
# from dataclasses import dataclass, field
# import numpy as np


# class MetadataFieldName(Enum):
#     DwaveSamplesetMetadata = "dwave_sampleset_metadata"
#     TimeMeasurements = "time_measurements"
#     DWaveSampleset = "dwave_sampleset"
#     Timing = "timing"
#     ProblemID = "problem_id"
#     ChainStrength = "chain_strength"
#     ChainBreakFraction = "chain_break_fraction"
#     ChainBreakMethod = "chain_break_method"
#     Embedding = "embedding"
#     Warnings = "warnings"


# @dataclass
# class ChainsData:
#     chain_strenght: float = field(default=None)
#     chain_break_fraction: float = field(default=None)


# @dataclass
# class HierarchicalRunMetadata:
#     dwave_sampleset_metadata: np.recarray = field(init=False)
#     time_measurements: np.recarray = field(init=False)
#     dwave_sampleset: dict = field(init=False)
#     timing: dict = field(init=False)
#     problem_id: str | int | float = field(init=False)
#     chain_strength: float = field(init=False)
#     chain_break_fraction: float = field(init=False)
#     chain_break_method: str = field(init=False)
#     embedding: dict = field(init=False)
#     warnings: dict = field(init=False)

#     def __init__(self, sampleset: list[SamplesetData]):
#         self.dwave_sampleset_metadata = self._process_samples(
#             sampleset, MetadataFieldName.DwaveSamplesetMetadata.value
#         )
#         self.time_measurements = self._process_samples(
#             sampleset, MetadataFieldName.TimeMeasurements.value
#         )
#         self.dwave_sampleset = self._process_samples(
#             sampleset, MetadataFieldName.DWaveSampleset.value
#         )
#         self.timing = self._process_samples(
#             sampleset, MetadataFieldName.Timing.value
#         )
#         self.problem_id = self._process_samples(
#             sampleset, MetadataFieldName.ProblemID.value
#         )
#         self.chain_strength = self._process_samples(
#             sampleset, MetadataFieldName.ChainStrength.value
#         )
#         self.chain_break_fraction = self._process_samples(
#             sampleset, MetadataFieldName.ChainBreakFraction.value
#         )
#         self.chain_break_method = self._process_samples(
#             sampleset, MetadataFieldName.ChainBreakMethod.value
#         )
#         self.embedding = self._process_samples(
#             sampleset, MetadataFieldName.Embedding.value
#         )
#         self.warnings = self._process_samples(
#             sampleset, MetadataFieldName.Warnings.value
#         )

#     def _process_samples(
#         self, sampleset: list[SamplesetData], field_name: str
#     ) -> np.recarray:
#         dtype = [getattr(sampleset[0], field_name)][0].dtype.descr
#         concatenated = np.concatenate(
#             [
#                 np.array([division_rec], dtype=dtype)
#                 for division_rec in [
#                     getattr(division, field_name) for division in sampleset
#                 ]
#             ]
#         ).view(np.recarray)
#         return concatenated

from enum import Enum
from QHyper.solvers.base import SamplesetData
from dataclasses import dataclass, field
import numpy as np
from typing import Union, List, Dict, Any


class MetadataFieldName(Enum):
    DwaveSamplesetMetadata = "dwave_sampleset_metadata"
    TimeMeasurements = "time_measurements"
    DWaveSampleset = "dwave_sampleset"
    Timing = "timing"
    ProblemID = "problem_id"
    ChainStrength = "chain_strength"
    ChainBreakFraction = "chain_break_fraction"
    ChainBreakMethod = "chain_break_method"
    Embedding = "embedding"
    Warnings = "warnings"


@dataclass
class ChainsData:
    chain_strength: float = field(default=None)
    chain_break_fraction: float = field(default=None)


@dataclass
class HierarchicalRunMetadata:
    dwave_sampleset_metadata: np.recarray = field(init=False)
    time_measurements: np.recarray = field(init=False)
    dwave_sampleset: List[Dict[str, Any]] = field(init=False)
    timing: List[Dict[str, Any]] = field(init=False)
    problem_id: List[Union[str, int, float]] = field(init=False)
    chain_strength: List[float] = field(init=False)
    chain_break_fraction: List[float] = field(init=False)
    chain_break_method: List[str] = field(init=False)
    embedding: List[Dict[str, Any]] = field(init=False)
    warnings: List[Dict[str, Any]] = field(init=False)

    def __init__(self, sampleset: List[SamplesetData]):
        self.dwave_sampleset_metadata = self._process_samples(
            sampleset, MetadataFieldName.DwaveSamplesetMetadata.value
        )
        self.time_measurements = self._process_samples(
            sampleset, MetadataFieldName.TimeMeasurements.value
        )
        self.dwave_sampleset = self._process_samples(
            sampleset, MetadataFieldName.DWaveSampleset.value
        )
        self.timing = self._process_samples(
            sampleset, MetadataFieldName.Timing.value
        )
        self.problem_id = self._process_samples(
            sampleset, MetadataFieldName.ProblemID.value
        )
        self.chain_strength = self._process_samples(
            sampleset, MetadataFieldName.ChainStrength.value
        )
        self.chain_break_fraction = self._process_samples(
            sampleset, MetadataFieldName.ChainBreakFraction.value
        )
        self.chain_break_method = self._process_samples(
            sampleset, MetadataFieldName.ChainBreakMethod.value
        )
        self.embedding = self._process_samples(
            sampleset, MetadataFieldName.Embedding.value
        )
        self.warnings = self._process_samples(
            sampleset, MetadataFieldName.Warnings.value
        )

    def _process_samples(
        self, sampleset: List[SamplesetData], field_name: str
    ) -> Any:
        """
        Process a list of SamplesetData and extract a given field.
        - Returns np.recarray if the field is a recarray
        - Returns a list of values (dict, int, float, str, etc.) otherwise
        """
        first_value = getattr(sampleset[0], field_name)

        if isinstance(first_value, np.recarray):
            dtype = first_value.dtype.descr
            concatenated = np.concatenate(
                [
                    np.array([getattr(division, field_name)], dtype=dtype)
                    for division in sampleset
                ]
            ).view(np.recarray)
            return concatenated
        else:
            # Works for dicts, numbers, strings, lists, etc.
            return [getattr(division, field_name) for division in sampleset]
