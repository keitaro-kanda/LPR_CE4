from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

from .label_objects import Label
from .read_arrays import read_array
from .read_tables import read_table
from .general_objects import StructureList

from ..utils.constants import PDS4_DATA_ROOT_ELEMENTS, PDS4_DATA_FILE_AREAS, PDS4_TABLE_TYPES
from ..utils.logging import logger_init

from ..extern import six

# Safe import of numpy (not required)
try:
    import numpy as np
except ImportError:
    np = None

# Initialize the logger
logger = logger_init()

#################################


def pds4_read(filename, quiet=False, lazy_load=False, no_scale=False, decode_strings=True, use_numpy=True):
    """ Reads PDS4 compliant data into a `StructureList`.

        Given a PDS4 label, reads the PDS4 data described in the label and
        associated label meta data into a `StructureList`, with each PDS4 data
        structure (e.g. Array_2D, Table_Binary, etc) as its own `Structure`. By
        default all data structures described in the label are immediately
        read into memory.

        Notes
        -----
        Header and Stream Text data structures are currently unsupported
        and will be skipped on read-in.

        Python 2 v. Python 3: Non-data strings (label, meta data, etc)  in
        Python 2 will be decoded to ``unicode`` and in Python 3 they will
        be decoded to ``str``. The return type of all data strings is
        controlled by *decode_strings*.

        Parameters
        ----------
        filename : str or unicode
            The filename, including full or relative path if necessary, of
            the PDS4 label describing the data.
        quiet : bool, optional
            Suppresses all info/warnings from being output.
        lazy_load : bool, optional
            If True, then the data of each PDS4 data structure will not be
            read-in to memory until the first attempt to access it. Defaults
            to False.
        no_scale : bool, optional
            If True, returned data will be exactly as written in the data file,
            ignoring offset or scaling values. Defaults to False.
        decode_strings : bool, optional
            If True, strings data types contained in the returned data will be
            decoded to the ``unicode`` type in Python 2, and to the ``str``
            type in Python 3. If False, leaves string types as byte strings.
            Defaults to True.
        use_numpy : bool, optional
            Returned data will be an ``np.ndarray`` and use NumPy data types.
            Defaults to True if NumPy is installed.

        Returns
        -------
        StructureList
            Contains PDS4 data `Structure`'s, each of which contains the data,
            the meta data and the label portion describing that data structure.
            `StructureList` can be treated/accessed/used like a ``dict`` or
            ``list``.

        Examples
        --------

        Below we document how to read data described by an example label
        which has two data structures, an Array_2D_Image and a Table_Binary.
        An outline of the label, including the array and a table with 3
        fields, is given.

        >>> struct_list = pds4_read('/path/to/Example_Label.xml')

        Example Label Outline::

           Array_2D_Image: unnamed
           Table_Binary: Observations
               Field: order
               Field: wavelength
               Group: unnamed
                   Field: pos_vector

        All below documentation assumes that the above outlined label,
        containing an array that does not have a name indicated in the label,
        and a table that has the name 'Observations' with 3 fields as shown,
        has been read-in.

        Accessing Example Structures:

            To access the data structures in `StructureList`, which is returned
            by `pds4_read()`, you may use any combination of ``dict``-like or
            ``list``-like access.

            >>> unnamed_array = struct_list[0]
            >>>              or struct_list['ARRAY_0']

            >>> obs_table = struct_list[1]
            >>>          or struct_list['Observations']

        Label or Structure Overview:

            To see a summary of the data structures, which for Arrays shows the
            type and dimensions of the array, and for Tables shows the type
            and number of fields, you may use the `StructureList.info()` method.
            Calling `Structure.info()` on a specific ``Structure`` instead will
            provide a more detailed summary, including all Fields for a table.

            >>> struct_list.info()
            >>> unnamed_array.info()
            >>> obs_table.info()

        Accessing Example Label data:

            To access the read-in data, as an array-like (either ``list``,
            ``array.array`` or ``np.ndarray``), you can use the data attribute for a
            PDS4 Array data structure, or the field() method to access a field
            for a table.

            >>> unnamed_array.data
            >>> obs_table.field('wavelength')
            >>> obs_table.field('pos_vector')

        Accessing Example Label meta data:

            You can access all meta data in the label for a given PDS4 data
            structure or field via the ``OrderedDict`` meta_data attribute. The
            below examples use the 'description' element.

            >>> unnamed_array.meta_data['description']

            >>> obs_table.field('wavelength').meta_data['description']
            >>> obs_table.field('pos_vector').meta_data['description']

        Accessing Example Label:

            The XML for a label is also accessible via the label attribute,
            either the entire label or for each PDS4 data structure.

            Entire label:
                >>> struct_list.label

            Part of label describing Observations table:
                >>> struct_list['Observations'].label
                >>> struct_list[1].label

            The returned object is similar to an ElementTree instance. It is
            searchable via `Label.find()` and `Label.findall()` methods and XPATH.
            Consult ``ElementTree`` manual for more details. For example,

            >>> struct_list.label.findall('.//disp:Display_Settings')

            Will find all elements in the entire label named 'Display_Settings'
            which are in the 'disp' prefix's namespace. You can additionally use the
            `Label.to_dict()` and `Label.to_string()` methods.
    """

    # Set logger to only emit error messages if requested
    if quiet:
        logger.quiet()

    # Set exception hook, which automatically calls logger on every uncaught exception
    sys.excepthook = _handle_exception

    # Initialize the log recording
    logger.get_handler('log_handler').begin_recording()

    # Disable use_numpy if NumPy is not installed
    if np is None:
        use_numpy = False

    # Read-in the PDS4 label
    logger.info('Processing label: ' + filename)

    label = Label()
    label.load(filename)

    # Read and extract all the PDS4 data structures specified in this label
    structures = read_structures(label, filename, lazy_load=lazy_load, no_scale=no_scale,
                                 decode_strings=decode_strings, use_numpy=use_numpy)

    # Save the log recording
    log = logger.get_handler('log_handler').get_recording(reset=False)

    # Create the structure list (which stores all the read-in structures,
    # the label and the log created while reading structures)
    structure_list = StructureList(structures, label, log)

    # Check if XML does not appear to describe a PDS4 label
    if 'Product' not in structure_list.type:
        raise ValueError('No viable/supported PDS4 label found.')

    # Warn if potentially expected but did not find a supported PDS4 data structure
    elif (structure_list.type in PDS4_DATA_ROOT_ELEMENTS) and (len(structure_list) == 0):
        logger.warning('No viable/supported PDS4 data structures found.')

    return structure_list


def read_structures(label, label_filename, lazy_load=False, no_scale=False, decode_strings=False, use_numpy=False):
    """ Reads PDS4 data structures described in label into a ``list`` of `Structure`'s.

    Parameters
    ----------
    label : Label
        Entire label of a PDS4 product.
    label_filename : str or unicode
        The filename, including full or relative path, of the label.
    lazy_load : bool, optional
        If True, does not read-in data of each data structure until the first attempt
        to access it. Defaults to False.
    no_scale : bool, optional
        If True, returned data will not be adjusted according to the offset and scaling
        factor. Defaults to False.
    decode_strings : bool, optional
        If True, strings data types contained in the returned data will be decoded to
        the ``unicode`` type in Python 2, and to the ``str`` type in Python 3. If
        false, leaves string types as byte strings. Defaults to False.
    use_numpy : bool, optional
        Returned data will be an ``np.ndarray`` and use NumPy data types. Defaults to False.

    Returns
    -------
    list[Structure]
        The PDS4 data `Structure`'s described in the label.
    """
    structures = []

    # Storage for number of structures types added
    num_structures = {'array': 0, 'table': 0}

    # The path of the data file is relative to the path of the label according to the PDS4 Standard
    data_path = os.path.dirname(label_filename)

    # Find all File Areas that can contain supported data structures (e.g. File_Area_Observation)
    file_areas = []

    for file_area_name in PDS4_DATA_FILE_AREAS:
        file_areas += label.findall(file_area_name)

    # Loop over each File Area
    for i, file_label in enumerate(file_areas):

        data_filename = file_label.findtext('.//file_name')
        data_filepath = os.path.join(data_path, data_filename)

        # Loop over each data structure inside the File Area
        for j, structure_label in enumerate(file_label):

            structures_xml = structure_label.getroot()

            # Determine structure is a table or an array
            if 'Array' in structures_xml.tag:
                structure_type = 'array'
            elif structures_xml.tag in PDS4_TABLE_TYPES:
                structure_type = 'table'
            else:
                # Skip over structure if its not actually a supported data structure
                continue

            # Increment structure type counter
            num_structures[structure_type] += 1

            # Create the structure
            args = [label, structure_label, data_filepath]
            kwargs = {'lazy_load': True, 'no_scale': no_scale, 'use_numpy': use_numpy}

            if structure_type == 'array':
                structure = read_array(*args, **kwargs)

            elif structure_type == 'table':
                structure = read_table(*args, decode_strings=decode_strings, **kwargs)

            # Set an ID for the structure if it has neither a local identifier or name in the label
            if structure.id is None:
                structure.id = '{0}_{1}'.format(structure_type.upper(), num_structures[structure_type] - 1)

            # Output that structure has been found
            if lazy_load:
                logger.info('Found a {0} structure: {1}'.format(structure.type, structure.id))

            # Attempt to access the data property such that the data gets read-in (if not on lazy-load)
            else:
                logger.info('Now processing a {0} structure: {1}'.format(structure.type, structure.id))
                structure.data

            structures.append(structure)

    return structures


def read_byte_data(data_filename, start_byte, stop_byte):
    """ Reads byte data from specified start byte to specified end byte.

    Notes
    -----
    This function will attempt to read until *stop_byte*, but if that is past the end-of-file it will
    silently read until end-of-file only.

    Parameters
    ----------
    data_filename : str or unicode
        Filename, including the full path, of the data file that contains the data for this structure.
    start_byte : int
        The start byte from which to begin reading.
    stop_byte : int
        The start byte at which to stop reading. May be -1, to specify read until EOF.

    Returns
    -------
    str or bytes
        The read-in byte data.
    """

    # Read in the data in binary form into byte_data
    try:

        # We use open() instead of io.open() because in Python 2.6, io.open() is subtly broken
        # under some circumstances. E.g., at least on Windows, it can take significant time to
        # read even small binary files
        with open(data_filename, 'rb') as file_handler:

            file_handler.seek(start_byte)

            # Read until stop byte
            if stop_byte > 0:
                read_size = stop_byte - start_byte
                byte_data = file_handler.read(read_size)

            # Read until EOF
            else:
                byte_data = file_handler.read()

        return byte_data

    except IOError as e:
        raise six.raise_from(IOError("Unable to read data from file '" + data_filename +
                                     "' found in label - {0}".format(e)), None)


def _handle_exception(exc_type, exc_value, exc_traceback):
    """
    Passed to sys.excepthook. This function logs all uncaught exceptions while reading the data
    (the exceptions will still be thrown).

    Parameters
    ----------
    exc_type
        The exception class
    exc_value
        The exception instance
    exc_traceback
        The exception traceback

    Returns
    -------
    None
    """

    logger.error(exc_value, exc_info=(exc_type, exc_value, exc_traceback))