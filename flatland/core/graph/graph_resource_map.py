from typing import Dict, Optional

from flatland.core.resource_map import ResourceMap


class GraphResourceMap(ResourceMap[str, str]):
    def __init__(self, _resource_map: Dict[str, str]):
        self._resource_map = _resource_map

    def get_resource(self, configuration: Optional[str]) -> Optional[str]:
        if configuration is None:
            return None
        return self._resource_map[configuration]
