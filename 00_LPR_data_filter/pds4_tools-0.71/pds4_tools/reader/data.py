from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import array

# Safe import of numpy (not required)
try:
    import numpy as np
except ImportError:
    np = None


def get_data_class(data, meta_data, use_numpy=False):
    """ Factory to create a `DataList`, `DataArray` or `DataNdarray` based on input.

    `pds4_read` is able to work both with and without NumPy. As such, it must be able to return
    at least two different forms of data, one of which uses a NumPy array (namely ``np.ndarray``)
    and one of which uses only Python's built-in ``array.array`` (which is used instead of ``list``
    for memory efficiency). Under some circumstances (see docstring of `DataList`) we must also
    use Python's built-in ``list``. Therefore we have 3 potential data types. For each data type
    we also want a meta data attribute, describing the meta data. The purpose of this method is
    to determine the type of data we want and create an object to hold it.

    Parameters
    ----------
    data : array_like
        PDS4 data.
    meta_data : Meta_Class
        Meta data for data.
    use_numpy : bool, optional
        If True then returned data object will be `DataNdarray`. Defaults to false.

    Returns
    -------
    `DataList`, `DataArray` or `DataNdarray`
        The appropriate data object for passed-in data.

    """

    if use_numpy:
        return DataNdarray(data, meta_data)

    elif isinstance(data, (list, tuple)):
        return DataList(data, meta_data)

    elif isinstance(data, array.array):
        return DataArray(data, meta_data)

    else:

        raise TypeError('Unknown type of Data found: {0}'.format(type(data)))


class DataList(list):
    """ Subclassed ``list`` that stores PDS4 data.

    Used only when use_numpy is False. Stores PDS4 string and complex data (neither of these data types
    can be handled well by ``array.array``), and integer data exceeding sizes available with ``array.array``.
    Will also store lists of `DataArray` as a proxy for multi-dimensional arrays (that is, nested DataLists
    are used when true multi-dimensional arrays of NumPy are not available), which can happen for arrays
    or for nested table fields.

    Parameters
    ----------
    data : list
        PDS4 data in the form of a list.
    meta_data : Meta_Class
        Meta data for the data.

    Attributes
    ----------
    meta_data: Meta_Class
        Meta data for the data
    """

    def __init__(self, data, meta_data):
        super(DataList, self).__init__(data)

        self.meta_data = meta_data

    def to_list(self):
        """ Obtain ``list`` in-place from `DataList`

        Returns a pure ``list``, without copying. It should very rare to ever need this because `DataList`
        allows for all things that ``list`` allows

        Returns
        -------
        list
        """
        return self[:]


class DataArray(array.array):
    """ Subclassed ``array.array`` that stores PDS4 data.

    Used only when use_numpy is False. Stores PDS4 numeric data, except for complex data and integer data
    exceeding sizes available with ``array.array``.

    Parameters
    ----------
    data : list
        PDS4 data in the form of a list.
    meta_data : Meta_Class
        Meta data for the data.

    Attributes
    ----------
    meta_data: Meta_Class
        Meta data for the data
    """

    def __new__(cls, data, meta_data):

        obj = array.array.__new__(cls, data.typecode, data)
        obj.meta_data = meta_data

        return obj

    def to_array(self):
        """ Obtain an ``array.array`` in-place from `DataArray`.

        Returns a pure ``array.array``, without copying. It should very rare to ever need this because
        `DataArray` allows for all things that ``array.array`` allows

        Returns
        -------
        array.array
        """

        return self[:]

# Try is used for this class declaration because numpy is not guaranteed to be available
try:

    class DataNdarray(np.ndarray):
        """ Subclassed ``np.ndarray`` that stores PDS4 data.

        Stores all types of PDS4 data when use_numpy is True.

        Parameters
        ----------
        data : list
            PDS4 data in the form of a list.
        meta_data : Meta_Class
            Meta data for the data.

        Attributes
        ----------
        meta_data: Meta_Class
            Meta data for the data.
        """

        def __new__(cls, data, meta_data):

            obj = np.asanyarray(data).view(cls)
            obj.meta_data = meta_data

            return obj

        def __array_finalize__(self, obj):

            if obj is None:
                return

            self.meta_data = getattr(obj, 'meta_data', None)

        def to_ndarray(self):
            """ Obtain an ``np.ndarray`` in-place from `DataNdarray`.

            Returns a pure ``np.ndarray``, without copying. It should very rare to ever need this because
            `DataNdarray` allows for all things that ``np.ndarray`` allows

            Returns
            -------
            np.ndarray
            """

            return self.view(np.ndarray)

# Define a dummy, unused, DataNdarray so it can be imported without error if numpy is unavailable
except AttributeError:

    class DataNdarray(object):

        def __init__(self, data, meta_data):
            pass

        pass

    pass
