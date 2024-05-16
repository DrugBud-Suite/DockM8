def parse_pocket_coordinates(pocket_arg):
    """
    Parses the pocket coordinates from the given pocket argument.

    The pocket argument should be in the format 'center:x,y,z*size:x,y,z',
    where 'center' represents the center coordinates of the pocket and 'size'
    represents the size of the pocket.

    Args:
        pocket_arg (str): The pocket argument to parse.

    Returns:
        dict or None: A dictionary containing the parsed pocket coordinates,
        where the keys are 'center' and 'size', and the values are lists of
        floats representing the coordinates.

    Raises:
        Exception: If there is an error parsing the pocket coordinates.

    """
    try:
        pocket_str = pocket_arg.split("*")
        pocket_coordinates = {}
        for item in pocket_str:
            key, value = item.split(":")
            pocket_coordinates[key] = list(map(float, value.split(",")))
    except Exception as e:
        print(
            f"Error parsing pocket coordinates: {e}. Make sure the pocket coordinates are in the format 'center:1,2,3*size:1,2,3'"
        )
        pocket_coordinates = None
    return pocket_coordinates
