from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from collections import Sequence

from .data import get_data_class
from .general_objects import Structure, Meta_Class, Meta_Structure

from ..utils.helpers import is_array_like, dict_extract
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


class TableStructure(Structure):
    """ Stores a single PDS4 table data structure.

    Contains the table's data, meta data and label portion. All forms of PDS4 tables
    (e.g. Table_Character, Table_Binary and Table_Delimited) are stored by this class.

    See `Structure`'s and `pds4_read`'s docstrings for attributes, properties and usage instructions
    of this object.

    Inherits all Attributes and Parameters from `Structure`. Overrides `info` method to implement it.
    """

    @classmethod
    def fromfile(cls, data_filename, structure_label, full_label,
                 lazy_load=False, no_scale=False, decode_strings=False, use_numpy=False):
        """ Create a table structure from relevant labels and file for the data.

        Parameters
        ----------
        data_filename : str or unicode
            Filename of the data file that contained the data for this table structure.
        structure_label : Label
            The segment of the label describing only this table structure.
        full_label : Label
            The entire label describing the PDS4 product this structure originated from.
        no_scale : bool, optional
            Read-in data will not be adjusted according to the offset and scaling factor. Defaults to False.
        decode_strings : bool, optional
            If True, strings data types contained in the returned data will be decoded to
            the ``unicode`` type in Python 2, and to the ``str`` type in Python 3. If
            false, leaves string types as byte strings. Defaults to False.
        lazy_load : bool, optional
            If True, does not read-in the data of this structure until the first attempt to access it.
            Defaults to False.
        use_numpy : bool, optional
            If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.

        Returns
        -------
        TableStructure
            An object representing the PDS4 table structure; contains its label, data and meta data.
        """

        # Create the meta data structure for this table
        meta_table_structure = Meta_TableStructure()
        meta_table_structure.load(structure_label)

        # Create the data structure for this table
        table_structure = cls(structure_data=None, structure_meta_data=meta_table_structure,
                              structure_label=structure_label, full_label=full_label,
                              parent_filename=data_filename)
        table_structure._use_numpy = use_numpy
        table_structure._no_scale = no_scale
        table_structure._decode_strings = decode_strings

        # Attempt to access the data property such that the data gets read-in (if not on lazy-load)
        if not lazy_load:
            table_structure.data

        return table_structure

    def info(self, abbreviated=False):
        """ Prints to stdout a summary of this data structure.

        Contains the type, number of fields and number of records in the Table.
        If *abbreviated* is True then also outputs the name and number of elements
        for each field in the table.

        Parameters
        ----------
        abbreviated : bool, optional
            If False, output additional detail. Defaults to False.

        Returns
        -------
        None
        """

        dimensions = self.meta_data.dimensions()

        abbreviated_info = "{0} '{1}' ({2} fields x {3} records)".format(
                           self.type, self.id, dimensions[0], dimensions[1])

        if abbreviated:
            print(abbreviated_info)

        else:
            print('Fields for {0}: \n'.format(abbreviated_info))

            for field in self.data:

                name = field.meta_data['name']
                full_name = field.meta_data.full_name(separator=', ')

                full_name_print = ''
                if name != full_name:
                    full_name_print = ' (full name: {0})'.format(full_name)

                print(name + full_name_print)

    def __getitem__(self, key):
        """ Get data for specific field in `TableStructure`.

        In the case of GROUP fields, the key can include the full location for disambiguation,
        (e.g. field_name = 'GROUP_1, GROUP_0, field_name'). See `info` method for full locations of all
        fields.

        Parameters
        ----------
        key : str, unicode, int, slice or tuple
            Selection for desired field. May be a string containing the name of a field to search for,
            similar to ``dict`` indexing functionality. May be an integer or slice specifying which field(s)'
            data to select,  similar to ``list`` or ``tuple`` indexing functionality. May be a two-valued
            tuple, with the first value providing the field name and the second value a zero-based repetition
            selector count if there are multiple fields with the same name.

        Returns
        -------
        DataList, DataArray, DataNdarray or None
            The data for the matched field, or None if no match found.

        Raises
        ------
        IndexError
            Raised if *key* is a larger int than there are fields in the table.

        Examples
        --------
        >>> table_struct[0]
        >>> table_struct['Field_Name']

        In-case of GROUP fields, we can use the full name (as given by the `info` method) for disambiguation,

        >>> table_struct['GROUP_1, GROUP_0, field_name']

        If both of the first two data structures have the name 'Field_Name', then to select the second
        we can do,

        >>> table_struct['Field_Name', 1]

        We can select both of the first two fields via,

        >>> table_struct[0:2]
        """

        if is_array_like(key):
            field = self.field(key[0], repetition=key[1])

        else:
            field = self.field(key)

        return field

    def __len__(self):
        """
        Returns
        -------
        int
            Number of fields contained in the table.
        """

        if self.data_loaded:
            num_fields = len(self.fields)

        else:
            dimensions = self.meta_data.dimensions()
            num_fields = dimensions[0]

        return num_fields

    def field(self, key, repetition=0, all=False):
        """ Get data for specific field in table.

        In the case of GROUP fields, the key can include the full location for disambiguation,
        (e.g. *key* = 'GROUP_1, GROUP_0, field_name'). See `info` method for full locations of all
        fields.

        Parameters
        ----------
        key : str, unicode, int or slice
            Selection for desired field. May be an integer or slice specifying which field(s)' data
            to select,  similar to ``list`` or ``tuple`` indexing functionality. May be a string
            containing the name of a field to search for.
        repetition : int, optional
            If there are multiple fields with the same name, specifies a zero-based repetition count
            to return. Defaults to 0.
        all : bool, optional
            If there are multiple fields with the same name, setting to True indicates that
            all fields should be returned in a ``list``. If set, and no match is found
            then an empty list will be returned. Defaults to False.

        Returns
        -------
        DataList, DataArray, DataNdarray, or list
            The data for the field(s).

        Raises
        ------
        IndexError
            Raised if *key* is a larger integer than there are fields in the table.
        KeyError
            Raised if *key* is a name that does not match any field.

        Examples
        --------
            See :func:`TableStructure.__getitem__` method examples.
        """

        return_fields = []

        # Search by index or slice
        if isinstance(key, six.integer_types) or isinstance(key, slice):
            return self.data[key]

        # Search by name
        for field in self.data:

            if field.meta_data['name'] == key:
                return_fields.append(field)

            elif field.meta_data.full_name(', ') == key:
                return_fields.append(field)

        # Return result
        if len(return_fields) > repetition and not all:
            return return_fields[repetition]

        elif all:
            return return_fields

        if repetition > 0:
            raise KeyError("Field '{0}' (repetition {1}) not found.".format(key, repetition))
        else:
            raise KeyError("Field '{0}' not found.".format(key))

    @threaded_cached_property
    def data(self):
        """ All data in the PDS4 table data structure.

        This property is implemented as a thread-safe cacheable attribute. Once it is run
        for the first time, it replaces itself with an attribute having the exact
        data that was originally returned.

        Unlike normal properties, this property/attribute is settable without a __set__ method.
        To never run the read-in routine inside this property, you need to manually create the
        the ``.data`` attribute prior to ever invoking this method (or pass in the data to the
        constructor on object instantiation, which does this for you).

        Returns
        -------
        list
            All the fields in this table.
        """

        from .read_tables import read_table_data
        read_table_data(self, no_scale=self._no_scale,
                        decode_strings=self._decode_strings, use_numpy=self._use_numpy)

        return self.data

    @property
    def fields(self):
        """
        Returns
        -------
        list
            All the fields in this table.
        """
        return self.data

    def add_field(self, data, meta_data, use_numpy=False):
        """ Add a field to the table.

        Parameters
        ----------
        data : array_like
            Data for the field to be added.
        meta_data : Meta_Field
            Meta data of the field to be added.
        use_numpy : bool, optional
            If True then added data will be cast to DataNdarray (effectively ```np.ndarray```) prior to
            saving. Defaults to False.

        Returns
        -------
        None
        """

        data = get_data_class(data, meta_data, use_numpy)

        if not self.data_loaded:
            self.data = []

        self.data.append(data)


class Meta_TableStructure(Meta_Structure):
    """ Meta data about a PDS4 table data structure.

    Meta data stored in this class is accessed in ``dict``-like fashion.  Stores meta data about all forms
    of Table (e.g. Table_Character, Table_Binary, and Table_Delimited). Normally this meta data originates
    from the label (e.g., if this is an Table_Character then everything from the opening tag of
    Table_Character to its closing tag will be stored in this object), via the `load` method.

    Inherits all Attributes, Parameters and Properties from `Meta_Structure`.

    Attributes
    ----------
    record : OrderedDict
        Convenience attribute for the Record_* portion of this table's meta data.
    type : str or unicode
        Type of table. One of 'Character', 'Binary', or 'Delimited'.

    Examples
    --------

    Supposing the following Table definition from a label::

        <Table_Binary>
          <local_identifier>data_Integration</local_identifier>
          ....
          <Record_Binary>
            <fields>9</fields>
            <groups>0</groups>
            <record_length unit="byte">65</record_length>
            <Field_Binary>
              <name>TIMESTAMP</name>
              ...
           ...
        ...
        </Table_Binary>

    >>> meta_table = Meta_TableStructure()
    >>> meta_table.load(structure_xml)

    >>> print(meta_table['local_identifier'])
    data_Integration

    >>> print(meta_table['Record_Binary']['record_length']
    65
    """

    def __init__(self, *args, **kwds):

        super(Meta_TableStructure, self).__init__(*args, **kwds)

        # Type of table (Character, Binary, Delimited)
        self.type = None

        # Record_* (Character, Binary, Delimited) of the table
        self.record = None

    def load(self, xml_table):
        """ Loads meta data into self from XML.

        Parameters
        ----------
        xml_table : Label or ElementTree Element
            Portion of label that defines the Table data structure.

        Returns
        -------
        None

        Raises
        ------
        PDS4StandardsException
            Raised if required meta data is absent.
        """

        self._load_keys_from_xml(xml_table)

        # Store record meta data with a variable name that is independent of the table-type
        record_str = [elem.tag for elem in xml_table if 'Record_' in elem.tag][0]
        self.type = record_str.split('_')[-1]
        self.record = self[record_str]

        # Ensure required keys for delimited tables exist
        if 'delimited' in self.type.lower():

            # Ensure required keys for table exist
            keys_must_exist = ['offset', 'records', 'record_delimiter', 'field_delimiter', record_str]
            self._check_keys_exist(keys_must_exist)

            # Ensure required keys for table's record exist
            keys_must_exist = ['fields', 'groups']
            self._check_keys_exist(keys_must_exist, sub_element=record_str)

        # Ensure required keys for fixed-width tables exist
        else:

            # Ensure required keys for table exist
            keys_must_exist = ['offset', 'records', record_str]
            self._check_keys_exist(keys_must_exist)

            # Ensure required keys for table's record exist
            keys_must_exist = ['fields', 'groups', 'record_length']
            self._check_keys_exist(keys_must_exist, sub_element=record_str)

    def dimensions(self):
        """ Obtains the number of fields and records for the table.

        Count does not include Group_Fields, but includes any non-group fields with-in them.

        Returns
        -------
        list
            Dimensions of the table, i.e., the number of fields and records, respectively.
        """

        normal_fields = sum(dict_extract(self.record, 'fields'))
        uni_sampled = 1 if ('Uniformly_Sampled' in self) else 0

        fields = normal_fields + uni_sampled
        records = self['records']

        return [fields, records]

    def is_fixed_width(self):
        """
        Returns
        -------
        bool
            True if the table is fixed-width (Table_Character and Table_Binary), false otherwise.
        """

        type = self.type.lower()

        if ('character' in type) or ('binary' in type):
            return True

        return False

    def is_delimited(self):
        """
        Returns
        -------
        bool
            True if the table is delimited (Table_Delimited), false otherwise.
        """

        if 'delimited' in self.type.lower():
            return True

        return False


class TableManifest(Sequence):
    """ Stores a single table's Meta_Fields and Meta_Groups

    The manifest is a representation the table, with fields and groups in the same order as they physically
    appear in the label, created by adding the appropriate Meta_Field and Meta_Group structures.

    Parameters
    ----------
    table_label : Label or ElementTree Element, optional
        Portion of label that defines the PDS4 table data structure. If given, the manifest is
        initialized from the label.
    """

    def __init__(self, table_label=None):

        self._struct = []
        self._table_type = None

        if table_label is not None:
            self.load(table_label)

        super(TableManifest, self).__init__()

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : int or slice
            Index of requested field or group.

        Returns
        -------
        Meta_Field or Meta_Group
            Matched field or group meta data.
        """
        return self._struct[key]

    def __len__(self):
        """
        Returns
        -------
        int
            Total number of fields and groups in the table.
        """
        return len(self._struct)

    def load(self, table_label):
        """ Initialize TableManifest with all appropriate Meta_Field's and Meta_Group's

        Parameters
        ----------
        table_label : Label or ElementTree Element
            Portion of label that defines the PDS4 table data structure.

        Returns
        -------
        None
        """

        # Find the <Record_*> and set the table type (e.g. Character, Binary or Delimited)
        record_str = [elem.tag for elem in table_label if 'Record_' in elem.tag][0]
        record_xml = table_label.find(record_str)
        self._table_type = record_str.split('_')[-1]

        # Add all fields and groups in <Record_*> to the TableManifest
        self._add_fields_and_groups(record_xml)

        # Set group names for all groups
        self._create_group_names()

        # Set location list for all fields and groups
        self._create_full_locations()

        # Perform basic sanity and basic validation checking on the table
        self._validate_table(record_xml)

    @property
    def num_items(self):
        """
        Returns
        -------
        int
            Total number of fields and groups in the table manifest.
        """
        return len(self)

    def fields(self):
        """
        Returns
        -------
        list[Meta_Field]
            All fields in the table manifest.
        """
        return [item for item in self if item.is_field()]

    def groups(self):
        """
        Returns
        -------
        list[Meta_Group]
            All groups in the table manifest.
        """
        return [item for item in self if item.is_group()]

    def get_field_by_full_name(self, field_full_name):
        """ Returns field(s) matched by full name.

        In the case of GROUP fields, the full name will include the full location for disambiguation,
        (e.g. field_full_name = 'GROUP_1, GROUP_0, field_name'). The full location includes the names of
        the groups necessary to reach this field.

        Parameters
        ----------
        field_full_name : str or unicode
            The full name of the field(s) to search for.

        Returns
        -------
        list[Meta_Field]
            List of matched fields.
        """
        matching_fields = [item for item in self._struct if item.full_name() == field_full_name]

        return matching_fields

    def get_children_by_idx(self, parent_idx, direct_only=False, return_idx=False):
        """ Obtains the child items (Meta_Fields and Meta_Groups) of the *parent_idx* item.

        Child items of a Group field are other Fields and Groups inside said Group field,
        and the children of those groups, etc.

        Parameters
        ----------
        parent_idx : int
            The index (in this TableManifest) of the Meta_Group whose children to find.
            A value of -1 indicates to find all items in the entire manifest.
        direct_only : bool, optional
            If True, only the immediate children of the *parent_idx* element are returned. Defaults to False.
        return_idx : bool, optional
            If True, changes the return type to a ``list`` containing the indicies (in this TableManifest)
            of the matched children. Defaults to False.

        Returns
        -------
        TableManifest
            A new table manifest containing only the matched children.
        """

        if return_idx:
            children = []
        else:
            children = TableManifest()

        if parent_idx == -1:
            parent_element_group_lvl = 0

        else:
            parent_element_group_lvl = self._struct[parent_idx].group_level + 1

        for i in range(parent_idx + 1, self.num_items):
            cur_element = self._struct[i]

            if cur_element.group_level >= parent_element_group_lvl:

                if cur_element.group_level == parent_element_group_lvl or not direct_only:
                    if return_idx:
                        children.append(i)
                    else:
                        children._append(cur_element)

            else:
                break

        return children

    def get_parent_by_idx(self, child_idx, return_idx=False):
        """ Obtains the parent Meta_Group of the child item given by *child_idx*.

        Parameters
        ----------
        child_idx : int
            The index (in this TableManifest) of the Meta_Field or Meta_Group whose parent to find.
        return_idx : bool, optional
            If True, changes the return type to an ``int``, which is the index (in this TableManifest) of the
            parent Meta_Group instead of the group itself. Returns -1 if a parent was not found. Defaults to
            False.

        Returns
        -------
        Meta_Group or None
            The parent of the specified child, or None if a parent was not found.
        """
        child_element = self._struct[child_idx]
        child_group_level = child_element.group_level

        for i in range(child_idx - 1, -1, -1):
            cur_element = self._struct[i]

            if (cur_element.group_level + 1) == child_group_level and cur_element.is_group():

                if return_idx:
                    return i
                else:
                    return cur_element

        if return_idx:
            return -1

        return None

    def _insert(self, key, field_or_group):
        """ Inserts an item (Meta_Field or Meta_Group) into the manifest.

        Parameters
        ----------
        key : int
            The key (index) the item will have in this TableManifest.
        field_or_group : Meta_Field or Meta_Group
            The field or group to insert.

        Returns
        -------
        None
        """

        self._struct.insert(key, field_or_group)

    def _append(self, field_or_group):
        """ Appends an item (Meta_Field or Meta_Group) at the end of the manifest.

        Parameters
        ----------
        field_or_group : Meta_Field or Meta_Group
            The field or group to append.

        Returns
        -------
        None
        """

        self._struct.append(field_or_group)

    def _add_fields_and_groups(self, xml_parent, group_level=0):
        """
        Adds all <Field_*>s and <Group_Field_*>s which are direct children of xml_parent to the
        TableManifest, and then recursively all children of the <Group_Field_*>s.

        The end result, when this is called on the <Record_*> of a table, is to add all Fields
        and all Groups defined inside <Record_*> into the manifest.

        Parameters
        ----------
        xml_parent : Label or ElementTree Element
            An XML element containing <Field_*> or <Group_Field_*> tags as children.
        group_level : int, optional
            The depth the current elements being added are nested from the <Record_*>. Defaults to 0,
            which means direct children. (Only non-zero for children of Group fields.)

        Returns
        -------
        None
        """
        field_tag = 'Field_' + self._table_type
        group_tag = 'Group_Field_' + self._table_type

        # Work around ElementTree not having ability to findall based on multiple conditions
        # while preserving order
        elements_xml = [element for element in xml_parent
                        if element.tag == field_tag or
                        element.tag == group_tag]

        # Iterate over all fields and groups
        for element in elements_xml:

            # Append Fields to Data
            if element.tag == field_tag:

                if self._table_type == 'Character':
                    appended_field = Meta_FieldCharacter()
                elif self._table_type == 'Binary':
                    appended_field = Meta_FieldBinary()
                elif self._table_type == 'Delimited':
                    appended_field = Meta_FieldDelimited()
                else:
                    raise ValueError('Unknown table type: ' + self._table_type)

                appended_field.load(element)
                self._append(appended_field)

                appended_field.group_level = group_level

            # Append Groups (and recursively sub-Fields and sub-Groups) to Data
            else:

                appended_group = Meta_Group()
                appended_group.load(element)
                self._append(appended_group)

                appended_group.group_level = group_level
                self._add_fields_and_groups(element, group_level + 1)

    def _create_group_names(self, group_level=0):
        """ Creates a name for each Meta_Group in the manifest that was not given one in the label.

        PDS4 Standards make group field names optional. This method creates names where necessary, with the
        format 'GROUP_n', where n is an integer from 0 to infinity (depending on the number of groups).
        In groups that are nested inside other groups, n will start from 0 regardless of the name of the
        parent group.

        Parameters
        ----------
        group_level : int, optional
            The depth the current group being named is nested from the <Record_*>. Defaults to 0,
            which means direct child.

        Returns
        -------
        None
        """
        group_indicies = [i for i, item in enumerate(self._struct)
                          if item.is_group()
                          if item.group_level == group_level]

        if not group_indicies:
            return

        prev_parent = -2
        counter = 0

        for group_idx in group_indicies:

            group = self._struct[group_idx]
            parent_idx = self.get_parent_by_idx(group_idx, return_idx=True)

            if parent_idx == prev_parent:
                counter += 1
            else:
                counter = 0

            if 'name' not in group:
                group['name'] = 'GROUP_' + six.text_type(counter)

            prev_parent = parent_idx

        self._create_group_names(group_level + 1)

    def _create_full_locations(self):
        """ Sets the correct full locations for all Meta_Fields and Meta_Groups in the manifest.

        A full location is a ``list`` of ``dict``s, where each ``dict`` has a single key and a single
        value. All together these keys and values indicate the full path to the table element from the
        beginning of the <Record_*>. Available keys are: 'field', 'group', and 'overlap'. A field key
        has a value corresponding to the name of a (non-group) field. A group key has a value corresponding
        to the name of a group field. An overlap has a value corresponding to the overlap number of the
        preceding field or group (zero-based index).

        Returns
        -------
        None

        Examples
        --------

        Suppose the following Example Label Outline::

            Table_Binary: Observations
                Field: order
                Field: wavelength           (1)
                Group: unnamed
                        Field: pos_vector   (2)
                Group: unnamed
                    Group: unnamed
                        Field: pos_vector
                        Field: pos_vector   (3)

        The full locations of the fields labeled above in parenthesis are as follows:

        (1) [{'field': 'wavelength'}]

        (2) [{'group': 'GROUP_0'},
             {'field': 'pos_vector'}]

        (3) [{'group': 'GROUP_1'},
             {'group': 'GROUP_0'},
             {'field': 'pos_vector'},
             {'overlap': '1'}]
        """

        for i, item in enumerate(self._struct):

            full_location = []

            # Begin by adding item name to full location
            idx = i

            type = 'group' if item.is_group() else 'field'
            full_location.insert(0, {type: item['name']})

            # Find _struct indicies of items having the same name and group location as current item
            identical_idxs = self._get_overlapping_item_idxs(i)

            # Add overlap count for groups and fields that have the same name and same group location
            # (otherwise these groups and fields would have exactly the same full name)
            if len(identical_idxs) >= 2:
                full_location.append({'overlap': identical_idxs.index(i)})

            parent_idx = self.get_parent_by_idx(idx, return_idx=True)

            # Add field parent group's full location
            # (this should already contain locations of all prior groups)
            if parent_idx != -1:
                group = self._struct[parent_idx]

                # Add parent group's location
                group_full_location = [location.copy() for location in group.full_location]
                group_full_location.extend(full_location)
                full_location = group_full_location

            item.full_location = full_location

    def _get_overlapping_item_idxs(self, item_idx):
        """
        Overlapping items are those Meta_Field's and Meta_Group's that have the same type (i.e., either
        field or group), name, and group location as `table_manifest[item_idx]`. The simplest case is two
        fields with the same name in a table without groups. To disambiguate these fields there is special
        handling, but first we need to find them, which is the purpose of this method.

        Parameters
        ----------
        item_idx : int
            The index (in this TableManifest) of the Meta_Field or Meta_Group
            whose overlapping items to find.

        Returns
        -------
        list[int]
            Indexes (in this TableManifest) of the Meta_Field's or Meta_Group's that overlap with `item_idx`.
        """
        item = self._struct[item_idx]
        item_parent_idx = self.get_parent_by_idx(item_idx, return_idx=True)

        overlapping_item_idxs = []

        # Determine if we're looking for overlapping fields or overlapping groups
        find_fields = True
        if item.is_group():
            find_fields = False

        child_idxs = self.get_children_by_idx(item_parent_idx, direct_only=True, return_idx=True)

        for idx in child_idxs:

            if self._struct[idx]['name'] == item['name']:

                if (find_fields and not self._struct[idx].is_group()) or (
                   not find_fields and self._struct[idx].is_group()):

                    overlapping_item_idxs.append(idx)

        return overlapping_item_idxs

    def _validate_table(self, record_xml):
        """
        Performs basic sanity checks on the table and basic verification of some PDS4 standards
        not currently tested by the schema and schematron.

        Checks performed:
            1) Ensure <fields> matches number of Field_* (including for nested fields)
            2) Ensure <groups> matches number of Group_Field_* (including for nested groups)
            3) Ensure the <format> of each field, if given, is valid according to the PDS4 standards
            4) Ensure that <group_repetitions> mod <group_length> equals zero for all group fields

        If any of the optional checks (1-3) fail, a warning is output if any of these fail.
        It is normally safe to continue. If an unsafe check fails, an exception is raised.

        Parameters
        ----------
        record_xml : Label or ElementTree Element
            The Record_* portion of the label for this table.

        Returns
        -------
        None

        Raises
        ------
        PDS4StandardsException
           Raised if group_repetitions do not evenly divide into group_length for all group fields.
        """

        # Check that number of fields and groups claimed matches number of <Field_*> and <Group_Field_*>
        num_rec_fields = int(record_xml.findtext('fields'))
        num_rec_groups = int(record_xml.findtext('groups'))

        groups = [item for item in self._struct if item.is_group() and item.group_level == 0]
        fields = [item for item in self._struct if not item.is_group() and item.group_level == 0]

        fields_mismatch_warn = '<fields> value does not match number of <Field_{0}> in'.format(self._table_type)
        groups_mismatch_warn = '<groups> value does not match number of <Group_Field_{0}> in'.format(self._table_type)

        if len(fields) != num_rec_fields:
            logger.warning('{0} Record_{1}.'.format(fields_mismatch_warn, self._table_type))

        if len(groups) != num_rec_groups:
            logger.warning('{0} Record_{1}.'.format(groups_mismatch_warn, self._table_type))

        for i, item in enumerate(self._struct):

            full_location_warn = "'{0}' (full location: '{1}')".format(item['name'], item.full_name(' -> '))

            if item.is_group():

                # Check number of fields and groups matching in nested groups
                children = self.get_children_by_idx(i, direct_only=True)

                groups = [child for child in children if child.is_group()]
                fields = [child for child in children if not child.is_group()]

                # For fixed-width tables, check that group_length is evenly divisible by group_repetitions
                if self._table_type != 'Delimited':

                    if len(fields) != item['fields']:
                        logger.warning(fields_mismatch_warn + ' group ' + full_location_warn)

                    if len(groups) != item['groups']:
                        logger.warning(groups_mismatch_warn + ' group ' + full_location_warn)

                    if item['length'] % item['repetitions'] != 0:
                        raise PDS4StandardsException("Group length '{0}' must be evenly divisible by the "
                                                     "number of repetitions '{1}' for group {2}"
                                                     .format(item['length'], item['repetitions'],
                                                             full_location_warn))

            elif 'format' in item:

                # Ensure field_format requirements
                valid_field_format = re.compile('^%([\+,-])?([0-9]+)(\.([0-9]+))?([doxfes])$')

                if not valid_field_format.match(item['format']):
                    logger.warning("field_format '{0}' does not conform to PDS4 standards for field {1} "
                                   .format(item['format'], full_location_warn))

                else:

                    is_numeric = {'ASCII_Real': True,
                                  'ASCII_Integer': True,
                                  'ASCII_NonNegative_Integer': True,
                                  'ASCII_Numeric_Base2': True,
                                  'ASCII_Numeric_Base8': True,
                                  'ASCII_Numeric_Base16': True}.get(item['data_type'], False)

                    sub_exp = valid_field_format.search(item['format']).groups()

                    plus_minus = sub_exp[0]
                    width = sub_exp[1]
                    precision = sub_exp[3]
                    specifier = sub_exp[4]

                    if plus_minus is not None:

                        if plus_minus == '+' and not is_numeric:
                            logger.warning("field_format '{0}' may not have '+' for a numeric data_type '{1}' for "
                                           "field {2}".format(item['format'], item['data_type'], full_location_warn))

                        elif plus_minus == '-' and is_numeric:
                            logger.warning("field_format '{0}' may not have '-' for a non-numeric data_type '{1}' for "
                                           "field {2}".format(item['format'], item['data_type'], full_location_warn))

                    if self._table_type == 'Character' and int(width) != item['length']:
                        logger.warning("in Table_Characters field_format '{0}' width must be equal to field_length "
                                       "'{1}' for field {2}".format(item['format'], item['length'], full_location_warn))

                    if precision is not None and specifier == 's' and int(width) != int(precision):
                        logger.warning("field_format '{0}' precision must be equal to width '{1}' for a string format "
                                       "for field {2}".format(item['format'], width, full_location_warn))


class Meta_TableElement(Meta_Class):
    """ Stores meta data about any table element.

    Table elements are Fields and Group fields. Subclassed by Meta_Field, Meta_Group and their derivatives.

    Meta data stored in this class is accessed in ``dict``-like fashion.  See docstring for `Meta_Class`
    for additional documentation.

    Parameters
    ----------
    (same as for `Meta_Class` and ``OrderedDict``)

    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    full_location : list[dict]
        See docstring for ``TableManifest._create_full_locations``.
    group_level : int
        The depth the element is nested from the <Record_*>. 0 means direct child.
    """

    def __init__(self, *args, **kwds):
        super(Meta_TableElement, self).__init__(*args, **kwds)

        self.full_location = None
        self.group_level = 0

    def load(self, element_xml):
        """ Initializes the meta data from an XML description of the element.

        Parameters
        ----------
        element_xml : Label or ElementTree Element
            Portion of the label describing this element.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Each type of `Meta_TableElement` subclassing this class must implement its own `load` method.
        """
        raise NotImplementedError

    def full_name(self, separator=', ', skip_parents=False):
        """ The full name of the element.

        The full name always includes an overlap disambiguator, to resolve two elements in the same
        location with the same name. Normally it also includes the name of any parent groups necessary
        to reach this element from the beginning of the record description.

        Parameters
        ----------
        separator : str, optional
            Used as the string separating each parent group and the name of the element. Defaults to ', '.
        skip_parents : bool, optional
            If True, parent groups of the element are excluded from the full name. Defaults to False.

        Returns
        -------
        str
            The name, including overlap disambiguator and parent groups if enabled, of the element.

        Examples
        --------

        Suppose the following Example Label Outline::

            Table_Binary: Observations
                Field: order
                Field: wavelength           (1)
                Group: unnamed
                        Field: pos_vector   (2)
                Group: unnamed
                    Group: example_group
                        Field: pos_vector
                        Field: pos_vector   (3)

        The full name of each element, with the default separator, is:

        (1) wavelength
        (2) GROUP_0, pos_vector
        (3) GROUP_1, example_group, pos_vector [1]

        Enabling skip_parents for element (3) would result in 'pos_vector [1]', which includes the
        overlap disambiguator, but excludes all parent groups.
        """

        # Attempt to begin extraction of name from full location
        full_name = None if (self.full_location is None) else ''

        # If full location is not set (should only be the case if this a fake field, for all fields
        # read from a real table full location should always be set), try to extract a name from
        # the name attribute
        if (full_name is None) and ('name' in self):
            full_name = self['name']

        # Build name by looping over the full location
        else:

            # If we are skipping outputting parent groups and fields, then find
            # the last index of a field or group in `self.full_location`
            if skip_parents:

                start_idx = None

                for i in range(len(self.full_location) - 1, -1, -1):

                    key = list(self.full_location[i].keys())[0]

                    if key in ['group', 'field']:
                        start_idx = i
                        break

            else:
                start_idx = 0

            # Location from which to start building the full name
            location = self.full_location[start_idx:]

            for i in range(0, len(location)):

                key = list(location[i].keys())[0]
                value = six.text_type(list(location[i].values())[0])

                # Add key/value to name
                if key == 'overlap':
                    full_name += ' [{0}]'.format(value)

                else:
                    full_name += value

                # Insert the separator where necessary (following group and field)
                if i < len(location) - 1:

                    next_key = list(location[i+1].keys())[0]

                    if next_key in ['group', 'field']:
                        full_name += separator

        return full_name

    def is_field(self):
        """
        Returns
        -------
        bool
            True if this element is a Field (i.e. Field_*), false otherwise.
        """

        return isinstance(self, Meta_Field)

    def is_group(self):
        """
        Returns
        -------
        bool
            True if this element is a Group field (i.e. Group_Field_*), false otherwise.
        """

        return isinstance(self, Meta_Group)


class Meta_Field(Meta_TableElement):
    """ Stores meta data about a single <Field_*>.

    Subclassed by `Meta_FieldCharacter`, `Meta_FieldBinary`, `Meta_FieldDelimited` and
    `Meta_FieldUniformlySampled`.

    Inherits ``full_location`` and ``group_level`` attributes from `Meta_TableElement`.

    Notes
    -----
    Keys starting with the 'field\_' prefix have this prefix removed.
    All other keys preserve their XML names.

    Examples
    --------

    Supposing the following Field definition from a label::

        <Field_Binary>
          <name>UTC</name>
          <field_location unit="byte">17</field_location>
          <data_type>ASCII_String</data_type>
          <field_length unit="byte">33</field_length>
          <unit>UTC date string</unit>
          <description>Time that the integration began</description>
        </Field_Binary>

    >>> field = Meta_Field()
    >>> field.load(field_xml)

    >>> print(field['name'])
    UTC

    >>> print(field['length'])
    33

    >>> print(field.keys())
    ['name', 'location', 'data_type', 'length', 'unit', 'description']
    """

    def load(self, field_xml):
        """ Initializes the meta data from an XML description of the Field.

        Parameters
        ----------
        field_xml : Label or ElementTree Element
            Portion of the label describing this Field.

        Returns
        -------
        None
        """

        self._load_keys_from_xml(field_xml, tag_modify=('field_', ''))
        self._check_keys_exist(['name'])


class Meta_FieldCharacter(Meta_Field):
    """ Stores meta data about a single <Field_Character>.

    Inherits ``full_location`` and ``group_level`` attributes from `Meta_Field`.

    See docstring of `Meta_Field` for usage information.
    """

    def load(self, field_xml):
        """ Initializes the meta data from an XML description of the element.

        Parameters
        ----------
        field_xml : Label or ElementTree Element
            Portion of the label describing this Field.

        Returns
        -------
        None
        """

        super(Meta_FieldCharacter, self).load(field_xml)

        keys_must_exist = ['location', 'length', 'data_type']
        self._check_keys_exist(keys_must_exist)


class Meta_FieldBinary(Meta_Field):
    """ Stores meta data about a single <Field_Binary>.

    Inherits ``full_location`` and ``group_level`` attributes from `Meta_Field`.

    See docstring of `Meta_Field` for usage information.
    """

    def load(self, field_xml):
        """ Initializes the meta data from an XML description of the Field.

        Parameters
        ----------
        field_xml : Label or ElementTree Element
            Portion of the label describing this Field.

        Returns
        -------
        None
        """

        super(Meta_FieldBinary, self).load(field_xml)

        keys_must_exist = ['location', 'length', 'data_type']
        self._check_keys_exist(keys_must_exist)

        # Not yet implemented
        # self.packed_fields = []
        # self._set_packed_data_fields(xml_field)


class Meta_FieldDelimited(Meta_Field):
    """ Stores meta data about a single <Field_Delimited>.

    Inherits ``full_location`` and ``group_level`` attributes from `Meta_Field`.

    See docstring of `Meta_Field` for usage information.
    """

    def load(self, field_xml):
        """ Initializes the meta data from an XML description of the Field.

        Parameters
        ----------
        field_xml : Label or ElementTree Element
            Portion of the label describing this Field.

        Returns
        -------
        None
        """

        super(Meta_FieldDelimited, self).load(field_xml)

        keys_must_exist = ['name', 'data_type']
        self._check_keys_exist(keys_must_exist)


class Meta_FieldUniformlySampled(Meta_Field):
    """ Stores meta data about a single <Uniformly_Sampled>.

    See docstring of `Meta_Field` for usage information.
    """

    def load(self, uniformly_sampled_xml):
        """ Initializes the meta data from an XML description of the Uniformly_Sampled.

        Parameters
        ----------
        uniformly_sampled_xml : Label or ElementTree Element
            Portion of the label describing this Uniformly Sampled field.

        Returns
        -------
        None
        """

        self._load_keys_from_xml(uniformly_sampled_xml, tag_modify=('sampling_parameter_', ''))
        self.full_location = [{'field': self['name']}]

        keys_must_exist = ['name', 'interval', 'first_value', 'last_value']
        self._check_keys_exist(keys_must_exist)


class Meta_FieldBit(Meta_Field):
    """ Stores meta data about a single <Field_Bit>. """

    def load(self, field_bit_xml):

        super(Meta_FieldBit, self).load(field_bit_xml)

        # Not yet implemented
        raise NotImplementedError


class Meta_Group(Meta_TableElement):
    """ Stores meta data about a single <Group_Field_*>.

    Notes
    -----
    Keys starting with the 'group\_' prefix have this prefix removed.
    All other keys preserve their XML names.

    Examples
    --------

    Supposing the following Group field definition from a label::

        <Group_Field_Binary>
          <repetitions>12</repetitions>
          <fields>1</fields>
          <groups>0</groups>
          <group_location unit="byte">1</group_location>
          <group_length unit="byte">24</group_length>
          <Field_Binary>
            ...
          </Field_Binary>
        </Group_Field_Binary>

    >>> group = Meta_Group()
    >>> group.load(field_xml)

    >>> print(group['repetitions'])
    12

    >>> print(group['location'])
    1

    >>> print(group.keys())
    ['repetitions', 'fields', 'groups', 'location', 'length', 'Field_Binary']
    """

    def load(self, group_field_xml):
        """ Initializes the meta data from an XML description of the Group_Field.

        Parameters
        ----------
        group_field_xml : Label or ElementTree Element
            Portion of the label describing this group field.

        Returns
        -------
        None
        """

        self._load_keys_from_xml(group_field_xml, tag_modify=('group_', ''))

        keys_must_exist = ['repetitions', 'fields', 'groups']
        if 'Delimited' not in group_field_xml.tag:
            keys_must_exist.extend(['location', 'length'])

        self._check_keys_exist(keys_must_exist)
