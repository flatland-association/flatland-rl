from typing import Generic, TypeVar, Optional

EntryPoint = TypeVar('EntryPoint')
Resource = TypeVar('Resource')


class ResourceMap(Generic[EntryPoint, Resource]):
    """
    Resource Map stores the single resource required to be at the entry point
    (i.e. to be in the cell or level-free crossing cell in grid world, resp. at the node in graph world).
    """

    def get_resource(self, entry_point: Optional[EntryPoint]) -> Optional[Resource]:
        raise NotImplementedError()
