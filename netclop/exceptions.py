"""Defines exceptions."""


class MissingResultError(Exception):
    """Exception raised when a result is missing."""
    def __init__(self, msg: str="Required result has not been produced.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class OverlappingPartitionError(Exception):
    """Exception raised when a result is missing."""
    def __init__(
        self,
        msg: str="Partition is overlapping (non-unique assignment of elements to parts).",
        *args,
        **kwargs
    ):
        super().__init__(msg, *args, **kwargs)
