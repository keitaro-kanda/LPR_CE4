from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import reduce

from .array_objects import ArrayStructure
from .data import get_data_class
from .data_types import (data_type_convert_array, get_binary_struct_type, adjust_complex_type,
                         apply_scaling_and_value_offset)

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


def _read_array_byte_data(array_structure):
    """ Reads the byte data from the data file for a PDS4 Array.

    Determines, from the structure's meta data, the relevant start and stop bytes in the data file prior to
    reading.

    Parameters
    ----------
    array_structure : ArrayStructure
        The PDS4 Array data structure for which the byte data needs to be read. Should have been
        initialized via `ArrayStructure.fromfile` method, or contain the required meta data.

    Returns
    -------
    str or bytes
        The exact byte data for the array.
    """

    from .core import read_byte_data

    meta_data = array_structure.meta_data

    num_elements = 1
    for axis_array in meta_data.get_axis_arrays():
        num_elements *= axis_array['elements']

    data_type = meta_data['Element_Array']['data_type']
    element_size = get_binary_struct_type(data_type)[3]

    start_byte = meta_data['offset']
    stop_byte = start_byte + num_elements * element_size

    return read_byte_data(array_structure.parent_filename, start_byte, stop_byte)


def flat_data_to_list(data, array_shape, element_offset=0):
    """
    Create a ``list`` having *array_shape*, with the values being sequentially extracted from
    flat array-like *data*.

    Parameters
    ----------
    data : array_like
        Flat array-like
    array_shape : array_like[int]
        Sequence of array dimensions.
    element_offset : int, optional
        The first element in the output ``list`` will be ``data[element_offset]``. Defaults to 0.

    Returns
    -------
    list
        The non-flat list with the same values as *data*.

    Examples
    --------

    >>> data = [int(7560*random.random()) for i in range(7560)]
    >>> array_shape = (21, 10, 36) # 21*10*36 = 7560
    >>> array_like = _flat_data_to_list(data, array_shape)

    >>> print np.asarray(array_like).shape
    (21, 10, 36)

    >>> array_like[2][5][7] == data[(2*10*36) + (5*36) + 7]
    True
    """

    data_list = []

    # Loop over each value in the first array dimension
    for i in range(0, array_shape[0]):

        # If array dimension has more than two values (i.e., the current array is at least a 3D array),
        # then recursively call _flat_data_to_list() on each element of this array, thereby
        # reducing the number of array dimensions with each recursive call
        if len(array_shape) > 2:

            # To actually take the correct values from flat data (which is done in the below else statement),
            # we need to know which element of data we should taking elements from. The below variable counts
            # how many array elements were already taken
            final_element_offset = element_offset + i * reduce(lambda x, y: x*y, array_shape[1:])

            # Reduce via recursion number of array dimensions from ndim = len(array_shape) to ndim-1
            reduced_array = flat_data_to_list(data, array_shape[1:], element_offset=final_element_offset)
            data_list.append(reduced_array)

        # Once we reach a 2D array then we can take the elements for that row from data and put them into
        # data_list
        else:

            # Calculate the index of the first element for this row
            num_elements = array_shape[1]
            final_element_offset = element_offset + i * num_elements

            # Extract from flat data only the datums for this specific row
            extracted_data_single = data[final_element_offset:final_element_offset+num_elements]

            # Append this row into data_list
            data_list.append(extracted_data_single)

    return data_list


def _apply_bitmask(data, bit_mask_string):
    """ Apply bitmask to *data*, modifying it in-place.

    Parameters
    ----------
    data : array_like
        Flat array-like integer data, byteswapped to be correct for endianness of current system if necessary
    bit_mask_string : str or unicode
        String of 1's and 0's, same length as number of bits in each *data* datum

    Returns
    -------
    None
    """

    # Skip needlessly applying bit_mask if it's all 1's
    if '0' not in bit_mask_string:
        return

    # Convert bit mask to binary (python assumes the input is a string describing the integer in MSB format,
    # which is what the PDS4 standard specifies.)
    bit_mask = int(bit_mask_string, 2)

    # Apply bit mask to each datum
    for i, datum in enumerate(data):
        data[i] = datum & bit_mask


def read_array_data(array_structure, no_scale, use_numpy):
    """
    Reads and properly formats the data for a single PDS4 array structure, modifies *array_structure* to
    contain all extracted fields for said table.

    Parameters
    ----------
    array_structure : ArrayStructure
        The PDS4 Array data structure to which the data should be added.
    no_scale : bool
        Returned data will not be adjusted according to the offset and scaling factor. Defaults to False.
    use_numpy : bool
        If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.

    Returns
    -------
    None
    """

    # Obtain the byte data of the array. It would save the extra memory we use at one point to both have the
    # byte data as a string and converted to its real data type if we used ``array.array.fromfile`` to read
    # the data into an array directly, but on some platforms (e.g. Windows) ``array.array`` does not support
    # 8 byte integers on Python 2 and Python <= 3.2, which are allowed by the PDS4 standard.
    array_byte_data = _read_array_byte_data(array_structure)

    # Obtain basic meta data
    meta_data = array_structure.meta_data
    element_array = meta_data['Element_Array']
    data_type = element_array['data_type']

    # Sort Axis Arrays by sequence number
    axis_arrays = meta_data.get_axis_arrays(sort=True)

    # Get the shape that extracted_data array-like will have
    array_shape = [axis_array['elements'] for axis_array in axis_arrays]

    # Convert byte data to its data_type
    converted_data = data_type_convert_array(data_type, array_byte_data, use_numpy)

    # Remove the byte data now that it is no longer needed to save memory
    del array_byte_data

    # Apply the bit mask to converted_data if necessary
    if ('Object_Statistics' in meta_data) and ('bit_mask' in meta_data['Object_Statistics']):
        datum_length = get_binary_struct_type(data_type)[3]
        bit_mask_string = six.text_type(meta_data['Object_Statistics']['bit_mask']).zfill(datum_length*8)
        _apply_bitmask(converted_data, bit_mask_string)

    # Make scaling_factor and value_offset adjustments
    scaling_factor = element_array.get('scaling_factor')
    value_offset = element_array.get('value_offset')

    # Adjust data values to account for 'scaling_factor' and 'value_offset' (in-place if possible)
    # (Note that this may change the data type to prevent overflow and thus also increase memory usage)
    if (not no_scale) and ((scaling_factor is not None) or (value_offset is not None)):
        converted_data = apply_scaling_and_value_offset(converted_data, scaling_factor, value_offset, use_numpy)

    # Adjust data to use complex data types if it had them
    if 'complex' in data_type.lower():
        converted_data = adjust_complex_type(converted_data, use_numpy)

    # Extract data from being flat into a properly formatted (nested) list representing the array
    extracted_data = converted_data
    if len(array_shape) > 1:
        extracted_data = flat_data_to_list(converted_data, array_shape)

    # Create the Data object
    extracted_data = get_data_class(extracted_data, meta_data, use_numpy)

    # Modify array_structure to contain the extracted and formatted data
    array_structure.data = extracted_data


def read_array(full_label, array_label, data_filename, lazy_load=False, no_scale=False, use_numpy=False):
    """ Create the `ArrayStructure`, containing label, data and meta data for a PDS4 Array.

    Used for all forms of PDS4 Arrays (e.g., Array, Array_2D_Image, Array_3D_Spectrum, etc).

    Parameters
    ----------
    full_label : Label
        The entire label for a PDS4 product, from which *array_label* originated.
    array_label : Label
        Portion of label that defines the PDS4 array data structure.
    data_filename : str or unicode
        Filename, including the full path, of the data file that contains the data for this array.
    lazy_load : bool, optional
        If True, does not read-in the data of this array until the first attempt to access it.
        Defaults to False.
    no_scale : bool, optional
        If True, returned data will not be adjusted according to the offset and scaling
        factor. Defaults to False.
    use_numpy : bool, optional
        If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.

    Returns
    -------
    ArrayStructure
        An object representing the array; contains its label, data and meta data

    Raises
    ------
    TypeError
        Raised if called on a non-array according to *array_label*.
    """

    # Skip over data structure if its not actually an Array
    if 'Array' not in array_label.tag:
        raise TypeError('Attempted to read_array() on a non-array: ' + array_label.tag)

    # Create the data structure for this array
    array_structure = ArrayStructure.fromfile(data_filename, array_label, full_label,
                                              lazy_load=True, no_scale=no_scale, use_numpy=use_numpy)

    return array_structure
