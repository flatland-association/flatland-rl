from typing import Dict, Optional

from flatland.core.resource_map import ResourceMap


class GraphResourceMap(ResourceMap[str, str]):
    def __init__(self, _resource_map: Dict[str, str]):
        self._resource_map = _resource_map

    def get_resource(self, from_entry_point: Optional[str], to_entry_point: Optional[str]) -> Optional[str]:
        if from_entry_point is None:
            return None
        return self._resource_map[from_entry_point]
