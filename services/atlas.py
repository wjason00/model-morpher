"""
Docstring for services.atlas
"""

import pandas as pd 
from brainglobe_atlasapi import BrainGlobeAtlas

class Atlas: 
    """
    Docstring for Atlas
    """
    def __init__(self, atlas_name: str):
        self.atlas = None 
        self.structures_df = None 

    def get_region_names(self) -> list:
        """
        Docstring for get_region_names

        :return: Description
        """

        return self.region_df['name'].tolist()
