"""
Docstring for services.atlas
"""

import pandas as pd 

from config import AVAILABLE_ATLASES
from brainglobe_atlasapi import BrainGlobeAtlas

class Atlas: 
    """
    Docstring for Atlas
    """

    def __init__(self):
        self.atlas = None 
        self.structures_df = None 

    
    def load_atlas(self, atlas_name: str):
        """
        Load a BrainGlobe atlas by name.
        """
        if atlas_name not in AVAILABLE_ATLASES:
            raise ValueError(f"Atlas '{atlas_name}' is not in the list of available atlases: {AVAILABLE_ATLASES}")
        
        self.atlas = BrainGlobeAtlas(atlas_name)

        # BrainGlobe already formats structures as a DataFrame - easy conversion
        self.structures_df = self.atlas.lookup_df[["acronym", "name"]].copy()

        return self.structures_df


    def search_structure(self, query: str) -> pd.DataFrame:
        """
        Search for structures by acronym or name.
        :param query: Substring to search for in acronyms or names.
        :return: DataFrame of matching structures.
        """
        if self.structures_df is None:
            raise RuntimeError("Atlas not loaded. Call load_atlas() before searching.")

        if not query:
            return self.structures_df

        # remove ws and case
        query = query.strip().lower()
        mask = (
            self.structures_df["acronym"].str.lower().str.contains(query) |
            self.structures_df["name"].str.lower().str.contains(query)
        )

        return self.structures_df[mask].reset_index(drop=True)
        

    def get_mesh_path(self, acronym: str) -> str:
        """
        Get the file path to the mesh for a given structure acronym.
        :param acronym: Structure acronym.
        :return: File path to the mesh.
        """
        if self.atlas is None:
            raise RuntimeError("Atlas not loaded. Call load_atlas() before getting mesh paths.")

        return str(self.atlas.meshfile_from_structure(acronym))
    
