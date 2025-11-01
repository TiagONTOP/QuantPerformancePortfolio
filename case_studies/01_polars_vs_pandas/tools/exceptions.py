"""
Custom exceptions for backtest validation.
"""


class InvalidParameterError(ValueError):
    """
    Raised when backtest parameters are invalid or inconsistent.

    Examples
    --------
    - Long threshold < short threshold
    - Negative transaction costs
    - Invalid Information Coefficient (not in [0, 1))
    - Non-positive window sizes
    """

    pass


