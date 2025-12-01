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
import pickle
import os
from dimod.sampleset import SampleSet
import json


class MetadataFieldName(Enum):
    DwaveSamplesetMetadata = "dwave_sampleset_metadata"
    TimeMeasurements = "time_measurements"
    DWaveSampleset = "dwave_sampleset"
    Timing = "timing"
    ProblemID = "problem_id"
    CommunityHash = "community_hash"
    ChainStrength = "chain_strength"
    ChainBreakFraction = "chain_break_fraction"
    ChainBreakMethod = "chain_break_method"
    Embedding = "embedding"
    Warnings = "warnings"
    Community = "community"


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
    community_hash: List[Union[str, int]] = field(init=False)
    chain_strength: List[float] = field(init=False)
    chain_break_fraction: List[float] = field(init=False)
    chain_break_method: List[str] = field(init=False)
    embedding: List[Dict[str, Any]] = field(init=False)
    warnings: List[Dict[str, Any]] = field(init=False)
    community: List[List[int]] = field(init=False)

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
        self.community_hash = self._process_samples(
            sampleset, MetadataFieldName.CommunityHash.value
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
        self.community = self._process_samples(
            sampleset, MetadataFieldName.Community.value
        )

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < self.__len__():
            result = SamplesetData(
                dwave_sampleset_metadata=self.dwave_sampleset_metadata[self.idx],
                time_measurements=self.time_measurements[self.idx],
                dwave_sampleset=self.dwave_sampleset[self.idx],
                timing=self.timing[self.idx],
                problem_id=self.problem_id[self.idx],
                community_hash=self.community_hash[self.idx],
                chain_strength=self.chain_strength[self.idx],
                chain_break_fraction=self.chain_break_fraction[self.idx],
                chain_break_method=self.chain_break_method[self.idx],
                embedding=self.embedding[self.idx],
                warnings=self.warnings[self.idx],
                community=self.community[self.idx],
            )
            self.idx += 1
            return result
        else:
            raise StopIteration
        
    def __len__(self):
        return len(self.community_hash)
    

    # def __getitem__(self, index):
    #     if index < 0 or index >= self.__len__():
    #         raise IndexError("Index out of range")
    #     return SamplesetData(
    #         dwave_sampleset_metadata=self.dwave_sampleset_metadata[index],
    #         time_measurements=self.time_measurements[index],
    #         dwave_sampleset=self.dwave_sampleset[index],
    #         timing=self.timing[index],
    #         problem_id=self.problem_id[index],
    #         community_hash=self.community_hash[index],
    #         chain_strength=self.chain_strength[index],
    #         chain_break_fraction=self.chain_break_fraction[index],
    #         chain_break_method=self.chain_break_method[index],
    #         embedding=self.embedding[index],
    #         warnings=self.warnings[index],
    #     )

    def __getitem__(self, community_hash: str | int):
        self_hashes = self.community_hash
        if community_hash not in self_hashes:
            raise KeyError(f"Index must be community_hash in this method. Community hash '{community_hash}' not found")
        index = self_hashes.index(community_hash)
        if index < 0 or index >= self.__len__():
            raise IndexError("Index is a community_hash. Index out of range")
        return SamplesetData(
            dwave_sampleset_metadata=self.dwave_sampleset_metadata[index],
            time_measurements=self.time_measurements[index],
            dwave_sampleset=self.dwave_sampleset[index],
            timing=self.timing[index],
            problem_id=self.problem_id[index],
            community_hash=self.community_hash[index],
            chain_strength=self.chain_strength[index],
            chain_break_fraction=self.chain_break_fraction[index],
            chain_break_method=self.chain_break_method[index],
            embedding=self.embedding[index],
            warnings=self.warnings[index],
            community=self.community[index],
        ) 
    
    def get_with_hash_id(self, community_hash: str | int) -> SamplesetData:
        self_hashes = self.community_hash
        if community_hash not in self_hashes:
            raise KeyError(f"Index must be community_hash in this method. Community hash '{community_hash}' not found")
        index = self_hashes.index(community_hash)
        if index < 0 or index >= self.__len__():
            raise IndexError("Index is a community_hash. Index out of range")
        return SamplesetData(
            dwave_sampleset_metadata=self.dwave_sampleset_metadata[index],
            time_measurements=self.time_measurements[index],
            dwave_sampleset=self.dwave_sampleset[index],
            timing=self.timing[index],
            problem_id=self.problem_id[index],
            community_hash=self.community_hash[index],
            chain_strength=self.chain_strength[index],
            chain_break_fraction=self.chain_break_fraction[index],
            chain_break_method=self.chain_break_method[index],
            embedding=self.embedding[index],
            warnings=self.warnings[index],
            community=self.community[index],
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
        

    def save_to_files(self, base_filename: str) -> None:
        """
        Save each metadata field to a separate .npy file.
        """
        # Safety check to avoid overwriting existing files
        headers_file = f"{base_filename}_headers.pkl"
        if os.path.exists(headers_file):
            raise FileExistsError(f"File '{headers_file}' already exists. The saving action is not recommended.")
        for field in MetadataFieldName:
            data_file = f"{base_filename}_{field.value}.pkl"
            if os.path.exists(data_file):
                raise FileExistsError(f"File '{data_file}' already exists. The saving action is not recommended.")        
        
        headers = [field.value for field in MetadataFieldName]
        with open(f"{base_filename}_headers.pkl", "wb") as f:
            pickle.dump(headers, f)
        for field in MetadataFieldName:
            data = getattr(self, field.value)
            with open(f"{base_filename}_{field.value}.pkl", "wb") as f:
                pickle.dump(data, f)

        def save_dwave_samplesets_serializables(path: str):
            dwave_samplesets_serializables = [ds.to_serializable() for ds in self.dwave_sampleset]
            with open(path, 'wb') as file:
                pickle.dump(dwave_samplesets_serializables, file, protocol=pickle.HIGHEST_PROTOCOL)

        def save_embeddings_dicts(path: str):
            with open(path, 'w') as f:
                json.dump(self.embedding, f, indent=4)
                
        save_dwave_samplesets_serializables(f"{base_filename}_{MetadataFieldName.DWaveSampleset.value}.pickle")
        save_embeddings_dicts(f"{base_filename}_{MetadataFieldName.Embedding.value}_dict.json")

    @staticmethod
    def load_from_files(base_filename: str) -> None:
        with open(f"{base_filename}_headers.pkl", "rb") as f:
            headers = pickle.load(f)
        hierarchical_metadata_new = HierarchicalRunMetadata.__new__(HierarchicalRunMetadata)
        for field in MetadataFieldName:
            try:
                with open(f"{base_filename}_{field.value}.pkl", "rb") as f:
                    data = pickle.load(f)
                setattr(hierarchical_metadata_new, field.value, data)
            except Exception as e:
                print(f"Could not load {field.value}: {e}")
                setattr(hierarchical_metadata_new, field.value, None)
        assert sorted(headers) == sorted([field.value for field in MetadataFieldName])

        with open(f"{base_filename}_{MetadataFieldName.DWaveSampleset.value}.pickle", "rb") as file:
            data = pickle.load(file)
        data = [SampleSet.from_serializable(ds) for ds in data]
        setattr(hierarchical_metadata_new, MetadataFieldName.DWaveSampleset.value, data)


        # try:
        with open(f"{base_filename}_{MetadataFieldName.Embedding.value}_dict.json", "rb") as file:
            data = json.load(file)
        # except Exception as e:
        #     data = []
        setattr(hierarchical_metadata_new, MetadataFieldName.Embedding.value, data)
        
        return hierarchical_metadata_new
    

# from dwave.embedding.transforms import EmbeddedStructure, embed_qubo, embed_bqm

# e = EmbeddedStructure(sampler.to_networkx_graph().edges(), emb[0])

# from QHyper.problems.community_detection import CommunityDetectionProblem, Network
# from QHyper.converter import Converter
# from dimod import BinaryQuadraticModel
# from QHyper.solvers.quantum_annealing.dwave.advantage import convert_qubo_keys

# problem = CommunityDetectionProblem(Network(G, community=c), one_hot_encoding=False)
# qubo = Converter.create_qubo(problem, [])
# qubo_terms, offset = convert_qubo_keys(qubo)
# bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)