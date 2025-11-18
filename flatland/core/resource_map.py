from typing import Generic, TypeVar

ConfigurationType = TypeVar('ConfigurationType')
ResourceType = TypeVar('ResourceType')


class ResourceMap(Generic[ConfigurationType, ResourceType]):
    """
    Resource Map stores the single resource required to be in the configuration
    (i.e. to be in the cell or level-free cell crossing in grid world, resp. on the edge in graph world).
    """

    def get_resource(self, configuration: ConfigurationType) -> ResourceType:
        raise NotImplementedError()
