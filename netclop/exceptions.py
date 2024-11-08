"""Defines exceptions."""


class MissingResultError(Exception):
    """Exception raised when a result is missing."""
    def __init__(self, msg: str="Required result has not been produced.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
