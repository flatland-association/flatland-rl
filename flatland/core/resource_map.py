from typing import Generic, TypeVar, Optional

EntryPointType = TypeVar('EntryPointType')
ResourceType = TypeVar('ResourceType')


class ResourceMap(Generic[EntryPointType, ResourceType]):
    """
    Resource Map stores the single resource required to be at the entry point
    (i.e. to be in the cell or level-free crossing cell in grid world, resp. at the node in graph world).
    """

    def get_resource(self, entry_point: Optional[EntryPointType]) -> Optional[ResourceType]:
        raise NotImplementedError()
