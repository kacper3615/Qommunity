from enum import Enum
from QHyper.solvers.base import SamplesetData
from dataclasses import dataclass, field
import numpy as np


class MetadataFieldName(Enum):
    DwaveSamplesetMetadata = "dwave_sampleset_metadata"
    TimeMeasurements = "time_measurements"


@dataclass
class HierarchicalRunMetadata:
    dwave_sampleset_metadata: np.recarray = field(init=False)
    time_measurements: np.recarray = field(init=False)

    def __init__(self, sampleset: list[SamplesetData]):
        self.dwave_sampleset_metadata = self._process_samples(
            sampleset, MetadataFieldName.DwaveSamplesetMetadata.value
        )
        self.time_measurements = self._process_samples(
            sampleset, MetadataFieldName.TimeMeasurements.value
        )

    def _process_samples(
        self, sampleset: list[SamplesetData], field_name: str
    ) -> np.recarray:
        dtype = [getattr(sampleset[0], field_name)][0].dtype.descr
        concatenated = np.concatenate(
            [
                np.array([division_rec], dtype=dtype)
                for division_rec in [
                    getattr(division, field_name) for division in sampleset
                ]
            ]
        ).view(np.recarray)
        return concatenated
