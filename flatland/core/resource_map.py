from typing import Generic, TypeVar, Optional

EntryPointT = TypeVar('EntryPointT')
ResourceT = TypeVar('ResourceT')


class ResourceMap(Generic[EntryPointT, ResourceT]):
    """
    ResourceT Map stores the single resource required to be at the entry point
    (i.e. to be in the cell or level-free crossing cell in grid world, resp. at the node in graph world).
    """

    def get_resource(self, entry_point: Optional[EntryPointT]) -> Optional[ResourceT]:
        raise NotImplementedError()
