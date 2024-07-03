from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .general_objects import Structure, Meta_Class, Meta_Structure
from .label_objects import get_display_settings_for_lid, get_spectral_characteristics_for_lid

from ..utils.constants import PDS4_NAMESPACES, PDS4_DATA_FILE_AREAS
from ..utils.exceptions import PDS4StandardsException

from ..extern import six
from ..extern.cached_property import threaded_cached_property

# Safe import of OrderedDict
try:
    from collections import OrderedDict
except ImportError:
    from ..extern.ordered_dict import OrderedDict


class ArrayStructure(Structure):
    """ Stores a single PDS4 array data structure.

    Contains the array's data, meta data and label portion. All forms of Array
    (e.g. Array, Array_2D, Array_3D_Image, etc) are stored by this class.

    See `Structure`'s and `pds4_read`'s docstrings for attributes, properties and usage instructions
    of this object.

    Inherits all Attributes, Parameters and Properties from `Structure`. Overrides `info` method to
    implement it.
    """

    @classmethod
    def fromfile(cls, data_filename, structure_label, full_label,
                 lazy_load=False, no_scale=False, use_numpy=False):
        """ Create an array structure from relevant labels and file for the data.

        Parameters
        ----------
        data_filename : str or unicode
            Filename of the data file that contained the data for this array structure.
        structure_label : Label
            The segment of the label describing only this array structure.
        full_label : Label
            The entire label describing the PDS4 product this structure originated from.
        lazy_load : bool, optional
            If True, does not read-in the data of this structure until the first attempt to access it.
            Defaults to False.
        no_scale : bool, optional
            Read-in data will not be adjusted according to the offset and scaling factor. Defaults to False.
        use_numpy : bool, optional
            If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.

        Returns
        -------
        ArrayStructure
            An object representing the PDS4 array structure; contains its label, data and meta data.
        """

        # Create the meta data structure for this array
        meta_array_structure = Meta_ArrayStructure()
        meta_array_structure.load(structure_label, full_label)

        # Create the data structure for this array
        array_structure = cls(structure_data=None, structure_meta_data=meta_array_structure,
                              structure_label=structure_label, full_label=full_label,
                              parent_filename=data_filename)
        array_structure._use_numpy = use_numpy
        array_structure._no_scale = no_scale

        # Attempt to access the data property such that the data gets read-in (if not on lazy-load)
        if not lazy_load:
            array_structure.data

        return array_structure

    def info(self, abbreviated=False):
        """ Prints to stdout a summary of this data structure.

        Contains the type and dimensions of the Array, and if *abbreviated* is False then
        also outputs the name and number of elements of each axis in the array.

        Parameters
        ----------
        abbreviated : bool, optional
            If False, output additional detail. Defaults to False.

        Returns
        -------
        None
        """

        dimensions = self.meta_data.dimensions()

        abbreviated_info = "{0} '{1}' ({2} axes, {3})".format(
            self.type, self.id, len(dimensions), ' x '.join(six.text_type(dim) for dim in dimensions))

        if abbreviated:
            print(abbreviated_info)

        else:
            print('Axes for {0}: \n'.format(abbreviated_info))

            for axis in self.meta_data.get_axis_arrays():
                print('{0} ({1} elements)'.format(axis['axis_name'], axis['elements']))

    @threaded_cached_property
    def data(self):
        """ All data in the PDS4 array data structure.

        This property is implemented as a thread-safe cacheable attribute. Once it is run
        for the first time, it replaces itself with an attribute having the exact
        data that was originally returned.

        Unlike normal properties, this property/attribute is settable without a __set__ method.
        To never run the read-in routine inside this property, you need to manually create the
        the ``.data`` attribute prior to ever invoking this method (or pass in the data to the
        constructor on object instantiation, which does this for you).

        Returns
        -------
        DataArray or DataNdarray
            The array described by this data structure.
        """

        from .read_arrays import read_array_data
        read_array_data(self, no_scale=self._no_scale, use_numpy=self._use_numpy)

        return self.data


class Meta_ArrayStructure(Meta_Structure):
    """ Meta data about a PDS4 array data structure.

    Meta data stored in this class is accessed in ``dict``-like fashion.  Stores meta data about all forms
    of Array (e.g. Array, Array_2D, Array_3D_Image, etc). Normally this meta data originates from the
    label (e.g., if this is an Array_2D then everything from the opening tag of Array_2D to its closing
    tag will be stored in this object), via the `load` method.

    Attributes
    ----------
    display_settings : Meta_DisplaySettings
        Meta data about the Display Settings for this array data structure.
    spectral_characteristics : Meta_SpectralCharacteristics
        Meta data about the Spectral Characteristics for this array data structure.

    Inherits all Attributes, Parameters and Properties from `Meta_Structure`.

    Examples
    --------

    Supposing the following Array definition from a label::

        <Array_3D_Spectrum>
          <local_identifier>data_Primary</local_identifier>
          ...
          <Axis_Array>
            <axis_name>Time</axis_name>
            <elements>21</elements>
            <sequence_number>1</sequence_number>
          </Axis_Array>
          ...
        </Array_3D_Spectrum>

    >>> meta_array = Meta_ArrayStructure()
    >>> meta_array.load(structure_xml)

    >>> print(meta_array['local_identifier'])
    data_Primary

    >>> print(meta_array['Axis_Array']['elements']
    21

    """

    def __init__(self, *args, **kwds):
        super(Meta_ArrayStructure, self).__init__(*args, **kwds)

        # Contains the Meta_DisplaySettings and Meta_SpectralCharacteristics for this Array structure,
        # if they exist in the label
        self.display_settings = None
        self.spectral_characteristics = None

    def load(self, xml_array, full_label):
        """ Loads meta data into self from XML.

        Parameters
        ----------
        xml_array : Label or ElementTree Element
            Portion of label that defines the Array data structure.
        full_label : Label or ElementTree Element
            The entire label from which xml_array originated.

        Returns
        -------
        None

        Raises
        ------
        PDS4StandardsException
            Raised if required meta data is absent.
        """

        self._load_keys_from_xml(xml_array)

        # Ensure required keys for Array_* exist
        keys_must_exist = ['offset', 'axes', 'Axis_Array', 'Element_Array']
        self._check_keys_exist(keys_must_exist)

        # Ensure required keys for Axis_Array(s) exist
        axis_keys_must_exist = ['axis_name', 'elements', 'sequence_number']
        multiple_axes = True if self.num_axes() > 1 else False
        self._check_keys_exist(axis_keys_must_exist, sub_element='Axis_Array', is_sequence=multiple_axes)

        # Ensure required keys for Element_Array exist
        self._check_keys_exist(['data_type'], sub_element='Element_Array')

        # Add the Meta_DisplaySettings and Meta_SpectralCharacteristics if they exist in the label
        if 'local_identifier' in self:

            local_identifier = six.text_type(self['local_identifier'])

            display_xml = get_display_settings_for_lid(local_identifier, full_label)
            if display_xml is not None:
                self.display_settings = Meta_DisplaySettings()
                self.display_settings.load(full_label, local_identifier)

            spectral_xml = get_spectral_characteristics_for_lid(local_identifier, full_label)
            if spectral_xml is not None:
                self.spectral_characteristics = Meta_SpectralCharacteristics()
                self.spectral_characteristics.load(full_label, local_identifier)

    def dimensions(self):
        """
        Returns
        -------
        list
            Dimensions of the array.
        """

        dimensions = [axis_array['elements'] for axis_array in self.get_axis_arrays(sort=True)]

        return dimensions

    def num_axes(self):
        """
        Returns
        -------
        int
            Number of axes/dimensions in the array.
        """

        return len(self.get_axis_arrays())

    def get_axis_arrays(self, sort=True):
        """ Convenience method to always obtain Axis_Arrays as a ``list``.

        Parameters
        ----------
        sort : bool, optional
            Sorts returned Axis Arrays by sequence_number if True. Defaults to True.

        Returns
        -------
        list
            List of ``OrderedDict``'s containing meta data about each Axis_Array.
        """

        axis_arrays = self['Axis_Array']

        if isinstance(axis_arrays, (list, tuple)):
            axis_arrays = list(axis_arrays)

        else:
            axis_arrays = [axis_arrays]

        if sort:
            axis_arrays = sorted(axis_arrays, key=lambda x: x['sequence_number'])

        return axis_arrays

    def get_axis_array(self, axis_name=None, sequence_number=None):
        """ Searches for a specific Axis_Array.

        Either *axis_name*, *sequence_number* or both must be specified.

        Parameters
        ----------
        axis_name : str or unicode, optional
            Searches for an Axis_Array with this name.
        sequence_number : int, optional
            Searches for an Axis_Array with this sequence number.

        Returns
        -------
        OrderedDict or None
            The matched Axis_Array, or None if no match was found.
        """

        axis_arrays = self.get_axis_arrays()

        retrieved_axis = None

        for axis in axis_arrays:

            # Find by both axis_name and sequence_number
            if (axis_name is not None) and (sequence_number is not None):

                if (six.text_type(axis['axis_name']) == axis_name) and (axis['sequence_number'] == sequence_number):
                    retrieved_axis = axis

            # Find by axis_name
            elif (axis_name is not None) and (six.text_type(axis['axis_name']) == axis_name):
                retrieved_axis = axis

            # Find by sequence_number
            elif (sequence_number is not None) and (axis['sequence_number'] == sequence_number):
                retrieved_axis = axis

        return retrieved_axis


class Meta_DisplaySettings(Meta_Class):
    """ Stores PDS4 Display Settings meta data for a single Array data structure.

    Meta data stored in this class is accessed in ``dict``-like fashion. Normally this meta data originates
    from the label (from the beginning of the Display_Settings tag, to its closing tag), via the `load`
    method.

    Attributes
    ----------
    valid : bool
        True if Display Settings conform to supported PDS4 Standards, False otherwise.
    """

    def __init__(self, *args, **kwds):
        super(Meta_DisplaySettings, self).__init__(*args, **kwds)

        # Set to True if Display Dictionary being used on load() is valid/supported
        self.valid = True

    def load(self, label, structure_lid):
        """ Loads meta data into self from XML.

        Parameters
        ----------
        label : Label or ElementTree Element
            The entire label for the PDS4 product containing the Display Settings.
        structure_lid : str or unicode
            The local_identifier for the Array data structure which uses the Display Settings.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            Raised if Display Settings do not exist for the specified *structure_lid*.
        PDS4StandardsException
            Raised if a data structure having the local_identifier *structure_lid* was not found.
        """

        display_settings = get_display_settings_for_lid(structure_lid, label)
        if display_settings is None:
            raise KeyError("No Display_Settings exist in label for local identifier '{0}'".
                           format(structure_lid))

        # Find all structures in the label
        found_structures = []

        for file_area_name in PDS4_DATA_FILE_AREAS:
            found_structures += label.findall('.//{0}/*'.format(file_area_name))

        # Find structure being referenced by structure_lid
        xml_structure = None

        for found_structure in found_structures:

            found_lid = found_structure.findtext('local_identifier')

            if (found_lid is not None) and (found_lid == structure_lid):
                xml_structure = found_structure

        if xml_structure is None:
            raise PDS4StandardsException("A Data Structure having the LID '{0}', specified in "
                                         "Display_Settings, was not found".format(structure_lid))

        tag_modify = ('{{{0}}}'.format(PDS4_NAMESPACES['disp']), '')
        self._load_keys_from_xml(display_settings, tag_modify=tag_modify)

        try:
            self.validate(xml_structure)
            self.valid = True

        # The display dictionary is not guaranteed to valid (unsupported older versions exist)
        except KeyError:
            self.valid = False

    def validate(self, xml_structure):
        """ Validates the Display Settings to conform to PDS4 Standards.

        Sets self.valid to False if Display Settings are invalid or unsupported.

        Parameters
        ----------
        xml_structure : Label or ElementTree Element
            Portion of label describing the Array data structure which uses the Display Settings

        Returns
        -------
        None
        """

        # Ensure required keys for Display_Settings exist
        keys_must_exist = ['Local_Internal_Reference', 'Display_Direction']
        self._check_keys_exist(keys_must_exist)

        # Ensure required keys for Local_Internal_Reference exist
        reference_keys_must_exist = ['local_identifier_reference', 'local_reference_type']
        self._check_keys_exist(reference_keys_must_exist, sub_element='Local_Internal_Reference')

        # Ensure required keys for Display_Direction exist
        display_keys_must_exist = ['horizontal_display_axis', 'horizontal_display_direction',
                                   'vertical_display_axis', 'vertical_display_direction']
        self._check_keys_exist(display_keys_must_exist, sub_element='Display_Direction')

        # Ensure required keys for Color_Display_Settings exists, if the class exists
        if 'Color_Display_Settings' in self:

            color_keys_must_exist = ['color_display_axis', 'red_channel_band', 'green_channel_band',
                                     'blue_channel_band']
            self._check_keys_exist(color_keys_must_exist, sub_element='Color_Display_Settings')

        # Ensure required key for Movie_Display_Settings exists, if the class exists
        if 'Movie_Display_Settings' in self:
            self._check_keys_exist(['time_display_axis'], sub_element='Movie_Display_Settings')

        # Ensure required axes referenced by the Display Dictionary actually exist in the structure
        axes_arrays = xml_structure.findall('Axis_Array')
        display_direction = self['Display_Direction']
        color_settings = self.get('Color_Display_Settings')
        movie_settings = self.get('Movie_Display_Settings')

        horizontal_axis_exists = False
        vertical_axis_exists = False
        color_axis_exists = False
        movie_axis_exists = False

        for axis in axes_arrays:

            axis_name = axis.findtext('axis_name')

            if axis_name == display_direction['horizontal_display_axis']:
                horizontal_axis_exists = True

            if axis_name == display_direction['vertical_display_axis']:
                vertical_axis_exists = True

            if (color_settings is not None) and (axis_name == color_settings['color_display_axis']):
                color_axis_exists = True

            if (movie_settings is not None) and (axis_name == movie_settings['time_display_axis']):
                movie_axis_exists = True

        display_axes_error = (not horizontal_axis_exists) or (not vertical_axis_exists)
        color_axis_error = (color_settings is not None) and (not color_axis_exists)
        movie_axis_error = (movie_settings is not None) and (not movie_axis_exists)

        if display_axes_error or color_axis_error or movie_axis_error:
            structure_lid = xml_structure.find('local_identifier')
            raise PDS4StandardsException("An axis_name, specified in the Display Dictionary for LID '{0}', "
                                         "was not found".format(structure_lid))


class Meta_SpectralCharacteristics(Meta_Class):
    """ Stores PDS4 Spectral Characteristics meta data for a single Array data structure.

    Meta data stored in this class is accessed in ``dict``-like fashion. Normally this meta data originates
    from the label (from the beginning of the Spectral_Characteristics tag, to its closing tag), via
    the `load` method.

    Attributes
    ----------
    valid : bool
        True if the Spectral Characteristics conform to supported PDS4 Standards, False otherwise.
    """

    def __init__(self, *args, **kwds):
        super(Meta_SpectralCharacteristics, self).__init__(*args, **kwds)

        # Set to True if Spectral Characteristics being used on load() is valid/supported
        self.valid = True

    def load(self, label, structure_lid):
        """ Loads meta data into self from XML.

        Parameters
        ----------
        label : Label or ElementTree Element
            The entire label for the PDS4 product containing the Spectral Characteristics.
        structure_lid : str or unicode
            The local_identifier for the Array data structure which uses the Spectral Characteristics.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            Raised if Spectral Characteristics do not exist for the specified *structure_lid*
        PDS4StandardsException
            Raised if a data structure having the local_identifier *structure_lid* was not found
        """

        spectral_chars = get_spectral_characteristics_for_lid(structure_lid, label)
        if spectral_chars is None:
            raise KeyError("No Spectral_Characteristics exist in label for local identifier '{0}'.".
                           format(structure_lid))

        # Find all structures in the label
        found_structures = []

        for file_area_name in PDS4_DATA_FILE_AREAS:
            found_structures += label.findall('.//{0}/*'.format(file_area_name))

        # Find structure being referenced by structure_lid
        xml_structure = None

        for found_structure in found_structures:

            found_lid = found_structure.findtext('local_identifier')

            if (found_lid is not None) and (found_lid == structure_lid):
                xml_structure = found_structure

        if xml_structure is None:
            raise PDS4StandardsException("A Data Structure having the LID '{0}', specified in "
                                         "Spectral_Characteristics, was not found".format(structure_lid))

        tag_modify = ('{{{0}}}'.format(PDS4_NAMESPACES['sp']), '')
        self._load_keys_from_xml(spectral_chars, tag_modify=tag_modify)

        try:
            self.validate(xml_structure)
            self.valid = True

        # The spectral characteristics class is not guaranteed to valid (unsupported older versions may exist)
        except KeyError:
            self.valid = False

    def validate(self, xml_structure):

        # We do not currently validate Spectral Characteristics since nothing is ever done with them
        # except display as label
        return

        # # structure_lid = xml_structure.find('local_identifier')
        #
        # # Ensure required keys for Spectral_Characteristics exist
        # keys_must_exist = ['Local_Internal_Reference', 'bin_width_desc']
        # self._check_keys_exist(keys_must_exist)
        #
        # # One of <Axis_Uniformly_Sampled>, <Axis_Bin_Set>, and <Spectral_Lookup> must exist
        # if self.get('Axis_Uniformly_Sampled') is None