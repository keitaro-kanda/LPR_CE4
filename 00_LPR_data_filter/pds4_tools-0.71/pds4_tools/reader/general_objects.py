from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Sequence

from ..utils.helpers import xml_to_dict, is_array_like
from ..utils.exceptions import PDS4StandardsException
from ..utils.logging import logger_init

from ..extern import six
from ..extern.cached_property import threaded_cached_property

# Safe import of OrderedDict
try:
    from collections import OrderedDict
except ImportError:
    from ..extern.ordered_dict import OrderedDict

# Initialize the logger
logger = logger_init()

#################################


class StructureList(Sequence):
    """ Stores the label and all supported data structures of a PDS4 product.

        An object of this type is returned by `pds4_read`. PDS4 supported data structures are forms of Arrays,
        Tables and Headers. This class allows both ``dict``-like and ``list``-like access to each individual
        PDS4 data structure inside.

        Parameters
        ----------
        structures : list[Structure]
            Each data `Structure`, including the data and the structure's label portion,
            in the PDS4 product.
        label : Label
            The entire label describing the PDS4 product.
        read_in_log : str or unicode
            Output of the log during read-in of the entire PDS4 product.

        Attributes
        ----------
        structures : list[Structure]
            Each data `Structure`, including the data and the structure's label portion,
            in the PDS4 product.
        label : Label
            The entire label describing the PDS4 product.
        read_in_log : str or unicode
            Output of the log during read-in of the entire PDS4 product.

        Examples
        --------

        Supposing the label described two objects, an Array_2D_Image (named 'Obs') and a Table_Binary
        (unnamed) in the same order as described here, they can be accessed as follows:

        >>> image_array = struct_list[0]
        >>>            or struct_list['Obs']

        >>> obs_table = struct_list[1]
        >>>          or struct_list['TABLE_0']

        See `pds4_read` and `__getitem__` docstrings for more examples.
    """

    def __init__(self, structures, label, read_in_log):
        super(StructureList, self).__init__()

        self.structures = structures

        self.label = label
        self.read_in_log = read_in_log

    def __getitem__(self, key):
        """ Searches `StructureList` for a specific data structure.

        Parameters
        ----------
        key : str, unicode, int, slice or tuple
            Selection for desired `Structure`. May be a string containing the name or local identifier of a
            single Structure, similar to ``dict`` indexing functionality.  May be an integer or slice
            specifying which Structure(s) to select, similar to ``list`` or ``tuple`` indexing functionality.
            May be a two-valued tuple, with the first value providing the name or local identifier and the
            second value a zero-based count, providing which repetition of Structure by that name to select.

        Returns
        -------
        Structure or list[Structure]
            Matched PDS4 data structure(s).

        Raises
        ------
        IndexError
            Raised if *key* is a larger integer than the number of Structures.
        KeyError
            Raised if *key* is a name or local identifier and does not match any Structure.

        Examples
        --------
        >>> struct_list[0]
        >>> struct_list['Observations']

        If both of the first two data structures have the name 'Observations', then to select the second
        we can do,

        >>> struct_list['Observations', 1]

        We can select both of the first two data structures via,

        >>> struct_list[0:2]
        """

        if isinstance(key, six.integer_types) or isinstance(key, slice):
            structure = self.structures[key]

        else:

            # Reuse array-search logic (where key is an array_like) for simple (ie, where key is a str)
            # name and lid searches
            if not is_array_like(key):
                key = (key,) + (0,)

            # Search by local identifier
            structure = self._get_structure_by_id(key, 'local_identifier')

            # Search by name
            if structure is None:
                structure = self._get_structure_by_id(key, 'name')

        if structure is None:

            if key[1] > 0:
                raise KeyError("Structure '{0}' (repetition {1}) not found.".format(key[0], key[1]))
            else:
                raise KeyError("Structure '{0}' not found.".format(key[0]))

        return structure

    def __len__(self):
        """
        Returns
        -------
        int
            Number of data structures contained.
        """
        return len(self.structures)

    def __repr__(self):
        """
        Returns
        -------
        str
            A repr string identifying the structure list, and all the structures it has.
        """

        string = '{0} with: \n{1}'.format(super(StructureList, self).__repr__(), repr(self.structures))

        return str(string)

    @property
    def type(self):
        """
        Examples of types include Product_Observational, Product_Ancillary, Product_Document, etc.

        Returns
        -------
        str or unicode
            Root tag of a PDS4 label.
        """

        return self.label.tag

    def info(self, abbreviated=True):
        """ Prints to stdout a summary of the contained data structures.

        For Arrays the summary contains the type and dimensions of the Array,
        and for Tables it contains the type and number of fields. Set *abbreviated*
        parameter to False to output additional detail.


        Parameters
        ----------
        abbreviated : bool, optional
            If False, output additional detail. Defaults to True.

        Returns
        -------
        None
        """

        for i, structure in enumerate(self.structures):

            if abbreviated:
                print('{0} : '.format(i), end='')
                structure.info(abbreviated=True)

            else:
                structure.info(abbreviated=False)
                print('---------------------------------------- \n')

    def _get_structure_by_id(self, key, id_type):
        """ Obtain a specific `Structure` from `StructureList` by an ID.

        Parameters
        ----------
        key : array_like[str or unicode, int]
            First value sets the key to search for (must be either the name or local identifier of the
            Structure), second value indicates which repetition to select, with zero-based indexing,
            typically used if there are multiple `Structure`'s with the same id.
        id_type : str or unicode
            Either 'name' or 'local identifier'.

        Returns
        -------
        Structure or None
            Matched PDS4 data structure, or None.
        """

        key = key[0], int(key[1])

        structure_lids = [obj.id for obj in self.structures]
        structure_match_idx = [i for i, lid in enumerate(structure_lids) if lid == key[0]]

        # If there is a match for repetition with either LID or name
        if len(structure_match_idx) > key[1]:
            structure = self.structures[structure_match_idx[key[1]]]

            # If there is a match for the specific id type requested
            if structure.meta_data.get(id_type) == key[0]:
                return structure

        return None


class Structure(object):
    """ Stores a single PDS4 data structure.

        Subclassed by `TableStructure` and `ArrayStructure`.

        Parameters
        ----------
        structure_data : DataNdarray, DataArray or DataList, optional
            The data in this PDS4 data structure. If not given and never set, data can be read-in
            via `fromfile`.
        structure_meta_data : Meta_Structure, optional
            Meta data describing this object (originating from the label).
        full_label : Label, optional
            The entire label describing the PDS4 product this structure originated from.
        structure_label : Label, optional
            The segment of the label describing only this data structure.
        parent_filename : str or unicode, optional
            Filename, including full path, of the data file that contained the data for this structure.

        Attributes
        ----------
        parent_filename : str or unicode
            Filename of the data file that contained the data for this structure.
        full_label : Label
            The entire label describing the PDS4 product this structure originated from.
        label : Label
            The segment of the label describing only this data structure.
        meta_data : Meta_Structure
            Meta data describing this object (originating from the label).
        data : DataNdarray, DataArray or DataList
            The data of this PDS4 data structure.

        Examples
        --------
        See `pds4_read` docstring for examples.
    """

    def __init__(self, structure_data=None, structure_meta_data=None, structure_label=None,
                 full_label=None, parent_filename=None):

        super(Structure, self).__init__()

        self._id = None if (structure_meta_data is None) else structure_meta_data.id

        self.parent_filename = parent_filename
        self.label = structure_label
        self.full_label = full_label
        self.meta_data = structure_meta_data

        # If data is given, set it. Otherwise the `data` method will lazy-load it as appropriate
        if structure_data is not None:
            self.data = structure_data

        # Controls whether data read-in via `fromfile` will use NumPy, whether it will be scaled and
        # whether byte strings will be decoded to unicode
        self._use_numpy = None
        self._no_scale = None
        self._decode_strings = None

    @property
    def id(self):
        """
        Returns
        -------
        str or unicode
            The ID (either local identifier if given, or name if given) of this data structure. If
            neither was given, an ID was likely assigned.
        """

        id = None

        if self._id:
            id = self._id

        elif self.meta_data:
            id = self.meta_data.id

        return id

    @id.setter
    def id(self, value):
        """
        Parameters
        ----------
        value : str or unicode
            The ID to set for this data structure.

        Returns
        -------
        None
        """
        self._id = value

    @property
    def type(self):
        """
        Returns
        -------
        str, unicode or None
            The official PDS4 data structure type name for this structure.
        """

        if self.label:
            return self.label.tag

        return None

    @property
    def data_loaded(self):
        """
        Returns
        -------
        bool
            True if the `data` attribute has been set (e.g. data has been read from file or set),
            False otherwise.
        """
        return 'data' in self.__dict__

    @classmethod
    def fromfile(cls, data_filename, structure_label, full_label,
                 lazy_load=False, no_scale=False, use_numpy=False):
        """ Create structure from relevant labels and file for the data.

        Parameters
        ----------
        data_filename : str or unicode
            Filename of the data file that contained the data for this structure.
        structure_label : Label
            The segment of the label describing only this data structure.
        full_label : Label
            The entire label describing the PDS4 product this structure originated from.
        lazy_load : bool, optional
            If True, does not read-in the data of this structure until the first attempt to access it.
            Defaults to False.
        no_scale : bool, optional
            Read-in data will not be adjusted according to the offset and scaling factor. Defaults to False.
        use_numpy : bool, optional
            If True, then data read-in, if not pass,

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Each type of `Structure` subclassing this class must implement its own `fromfile` method.
        """
        return NotImplementedError

    def info(self, abbreviated=False):
        """ Prints to stdout a summary of this data structure.

        Parameters
        ----------
        abbreviated : bool, optional
            If False, output additional detail. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Each type of `Structure` subclassing class must implement its own `info` method.
        """

        raise NotImplementedError

    @threaded_cached_property
    def data(self):
        """ The data of this PDS4 structure.

        Raises
        ------
        NotImplementedError
            Each type of `Structure` subclassing this class must implement its own `data` method.
        """
        raise NotImplementedError

    def is_array(self):
        """
        Returns
        -------
        bool
            True if this `Structure` is a form of a PDS4 array, false otherwise.
        """

        from .array_objects import ArrayStructure

        return isinstance(self, ArrayStructure)

    def is_table(self):
        """
        Returns
        -------
        bool
            True if this `Structure` is a form of a PDS4 table, false otherwise.
        """

        from .table_objects import TableStructure

        return isinstance(self, TableStructure)


class Meta_Class(OrderedDict):
    """ Contains meta data about any type of data.

    Subclassed by all other Meta_* classes, subclasses ``OrderedDict``. Most PDS4 meta data originates
    from the label, therefore we need a consistent interface to pull this meta data from a `Label` or
    ``ElementTree`` Element into an ``OrderedDict``, which any `Meta_Class` ultimately subclasses.

    Most often we do not use the actual ``OrderedDict`` constructor to populate the meta data, but
    instead use methods provided by this class to load the dictionary with keys directly from the XML.

    Meta data stored in this class is accessed in ``dict``-like fashion.

    Parameters
    ----------
    (same as for ``OrderedDict``)

    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    # Element tags that have text values which are always expected to be strings. Values of these
    # elements will not be cast from a string by default, even if said values contain an int or float.
    _DEFAULT_CAST_IGNORE = ('local_identifier', 'name', 'title', 'description')

    def __eq__(self, other):
        """Override the default equality test.

        Notes
        -----
        ``OrderedDict``, from which this method inherits, compares only key/value pairs for equality.
        However this does not take into account any attributes the meta classes may have or type equality.
        It is possible to implement real rich comparison but this does not appear necessary or useful;
        therefore we revert to the default id comparison.
        """
        return self is other

    def __ne__(self, other):
        """Override default non-equality test."""
        return not self == other

    def _load_keys_from_xml(self, xml, cast_values=True, cast_ignore=_DEFAULT_CAST_IGNORE, tag_modify=()):
        """ Loads keys into self from XML.

        Parameters
        ----------
        xml : Label or ElementTree Element
            XML from which to take keys.
        cast_values : bool, optional
            Casts values of contents while loading (e.g. a string 1 in the XML will be the int 1).
            Defaults to True.
        cast_ignore : tuple[str or unicode], optional
            If given, then a tuple of element tags. If *cast_values* is True, then for elements with
            tags matching exactly the values in this tuple, element values will not be cast. If
            *tag_modify* is set, then tags and attribute names specified by *cast_ignore* should be
            the already tag modified versions. Defaults to `_DEFAULT_CAST_IGNORE`.
        tag_modify : tuple, optional
            2-valued tuple with str or unicode elements, or tuple of 2-valued tuples. See description in
            `xml_to_dict`.

        Returns
        -------
        None
        """

        items = list(xml_to_dict(xml, skip_attributes=True, cast_values=cast_values,
                                 cast_ignore=cast_ignore, tag_modify=tag_modify).values())[0]

        for key, value in six.iteritems(items):
            self[key] = value

    def _check_keys_exist(self, keys, sub_element=None, is_sequence=False):
        """ Checks if keys exist in self.

        Parameters
        ----------
        keys : array_like
            Keys to check for existence.
        sub_element : str or unicode, optional
            If set then self[sub_element] is checked for keys instead of self.
        is_sequence : bool
            Must be used in conjunction with sub_element. If set then self[sub_element] is assumed to be a
            ``sequence``, and each element is checked for keys.

        Returns
        -------
        None

        Raises
        ------
        PDS4StandardsException
            Raised if a key does not exist.
        """

        # Determine where to look for keys
        if sub_element is not None:
            struct = self[sub_element]
            error_addition = "'{0}' in ".format(sub_element)
        else:
            struct = self
            error_addition = ''

        # Check for the existence of each key
        for key in keys:

            error = "{0}{1} must have the XML attribute '{2}'".format(error_addition, type(self).__name__, key)

            if is_sequence:

                # Recurse into struct if it's a sequence
                for element in struct:

                    if key not in element:
                        raise PDS4StandardsException(error)
            else:

                if key not in struct:
                    raise PDS4StandardsException(error)


class Meta_Structure(Meta_Class):
    """ Meta data about a PDS4 Data Structure.

    Meta data stored in this class is accessed in dict-like fashion. Normally this meta data
    originates from the label.

    Subclassed by a meta class for each data structure type.
    """

    @property
    def id(self):
        """
        Returns
        -------
        str or unicode
            The local_identifier of the PDS4 data structure if it exists, otherwise the name if it exists.
            If neither was specified in the label, None is returned.
        """
        id = self.get('local_identifier')

        if id is None:
            id = self.get('name')

        return id
