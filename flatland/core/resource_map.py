from typing import Generic, TypeVar, Optional

EntryPointT = TypeVar('EntryPointT')
ResourceT = TypeVar('ResourceT')


class ResourceMap(Generic[EntryPointT, ResourceT]):
    """
    ResourceT Map stores the single resource occupied while traversing the edge from `from_entry_point` to
    `to_entry_point` - the resource held is always the one at `from_entry_point` (the cell/node entered;
    `to_entry_point` is the neighbor entered once the edge is left, see `GraphTransitionMap`'s docstring
    for the `[u,v)` framing this mirrors), i.e. to be in the cell or level-free crossing cell in grid
    world, resp. at the node in graph world. Keyed by the edge rather than a bare entry point, so a
    resource can in principle depend on the whole transition, not just where it starts.
    """

    def get_resource(self, from_entry_point: Optional[EntryPointT], to_entry_point: Optional[EntryPointT]) -> Optional[ResourceT]:
        raise NotImplementedError()
