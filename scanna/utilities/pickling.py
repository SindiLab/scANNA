"""Utility functions related to data IO using pickle."""
import pickle


def Pickler(data, filename: str):
    """ A convenient utility function for compressing ("pickling") data.

    Args:
        data: the data source we want to compress.
        filename: the full filename (including path) of where we want to save
          the pickled file to.

    Returns:
        None.

    Raises:
        None.
    """

    with open(filename, 'wb+') as outfile:
        pickle.dump(data, outfile)


def Unpickler(filename: str):
    """ A convenient utility function for decompressing ("unpickling") data.

    Args:
        filename: the full filename (including path) of where we want to save
          the pickled file to.

    Returns:
        Decompressed version of the pickle file that was passed on to the
        function.

    Raises:
        None.
    """

    with open(filename, 'rb+') as infile:
        return_file = pickle.load(infile)
    return return_file
