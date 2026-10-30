"""Independent copies of large, trusted native-test graphs."""

import pickle


def copy_graph(graph):
    """Copy every value without Python recursion over millions of scalars.

    The bytes are produced here from test-owned in-memory data. Never accept
    serialized input from files or external sources at this boundary.
    """
    return pickle.loads(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))  # noqa: S301


def first_difference(left, right, path=""):
    """Compare every graph value, ignoring only SDK archive bookkeeping.

    Equal scalar arrays need no recursive calls or per-element path strings.
    Self-equality retains the original rejection of NaN, even if lists share
    the same NaN object. Containers still recurse with the metadata rules.
    """
    if isinstance(left, dict):
        for key in left:
            if key not in {"index", "size", "offset"} and (
                result := first_difference(left[key], right[key], path + "/" + key)
            ):
                return result
    elif isinstance(left, list):
        if len(left) != len(right):
            return path, len(left), len(right)
        if left == right and all(
            not isinstance(value, (dict, list)) and value == value for value in left  # noqa: PLR0124 - NaN
        ):
            return None
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if result := first_difference(a, b, path + "/" + str(index)):
                return result
    elif left != right:
        return path, left, right
    return None
