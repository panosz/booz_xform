def find_y_axis_crossing(x1, x2, y1, y2):
    """
    Find where the line passing through (`x1`, `y1`) and (`x2`, `y2`) crosses
    the y-axis.

    Multiple lines can be considered if `y1` and `y2` are array like.

    Parameters:
    -----------

    x1, x2: float,
        The x coordinates

    y1, y2: float or array like, shape(N)
        The y coordinates

    Returns:
    --------

    y0: float or array like, shape(N)
        The y coordinates of the points where the line/lines cross the y-axis.
    """

    return (y1*x2 - y2*x1)/(x2 - x1)
