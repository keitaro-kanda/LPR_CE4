from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
import sys
from array import array as arr

from ..utils.exceptions import PDS4StandardsException
from ..utils.logging import logger_init

from ..extern.six.moves import builtins

# Safe import of numpy (not required)
try:
    import numpy as np
except ImportError:
    np = None

# Initialize the logger
logger = logger_init()

#################################


def get_binary_struct_type(data_type):
    """ Obtain appropriate Python data types for a PDS4 binary data type.

    Parameters
    ----------
    data_type : str or unicode
        PDS4 data type.

    Returns
    -------
    tuple
        4 valued tuple of (byte-order, ``array.array`` typecode, numpy typecode, size in bytes).
    """

    struct_type = {'IEEE754MSBSingle': ('big', 'f', 'float32', 4),
                   'IEEE754MSBDouble': ('big', 'd', 'float64', 8),
                   'SignedMSB2': ('big', 'h', 'int16', 2),
                   'SignedMSB4': ('big', 'i', 'int32', 4),
                   'SignedMSB8': ('big', 'l', 'int64', 8),
                   'UnsignedMSB2': ('big', 'H', 'uint16', 2),
                   'UnsignedMSB4': ('big', 'I', 'uint32', 4),
                   'UnsignedMSB8': ('big', 'L', 'uint64', 8),
                   'ComplexMSB8': ('big', 'f', 'float32', 8),
                   'ComplexMSB16': ('big', 'd', 'float64', 16),

                   'IEEE754LSBSingle': ('little', 'f', 'float32', 4),
                   'IEEE754LSBDouble': ('little', 'd', 'float64', 8),
                   'SignedLSB2': ('little', 'h', 'int16', 2),
                   'SignedLSB4': ('little', 'i', 'int32', 4),
                   'SignedLSB8': ('little', 'l', 'int64', 8),
                   'UnsignedLSB2': ('little', 'H', 'uint16', 2),
                   'UnsignedLSB4': ('little', 'I', 'uint32', 4),
                   'UnsignedLSB8': ('little', 'L', 'uint64', 8),
                   'ComplexLSB8': ('little', 'f', 'float32', 8),
                   'ComplexLSB16': ('little', 'd', 'float64', 16),

                   'SignedByte': ('little', 'b', 'int8', 1),
                   'UnsignedByte': ('little', 'B', 'uint8', 1)
                   # 'SignedBitString': ('', '', '', None),   # Currently unhandled
                   # 'UnsignedBitString': ('', '', '', None)

                   }.get(data_type, None)

    # On Windows (and possibly some uncommon platforms) 'l' and 'L' in ``array.array`` have a max size of
    # 4 bytes instead of expected 8. We check for this, check if an alternate typecode is available,
    # otherwise return None for that typecode.
    if (struct_type is not None) and (struct_type[1] in ('l', 'L')):
        struct_type = (struct_type[0], _long_long_typecode(struct_type[1]), struct_type[2], struct_type[3])

    return struct_type


def get_min_integer_struct_type(data, use_numpy=False):
    """ Obtain smallest integer data type that can store every value in *data*.

    *data* must contain only integers.

    Parameters
    ----------
    data : array_like
        PDS4 integer data.
    use_numpy : bool, optional
        Setting to True if *data* is some form of NumPy array speeds up calculation. Defaults to False.

    Returns
    -------
    tuple
        2 valued tuple of (``array.array`` integer typecode, numpy integer typecode). Returns None for
        a typecode if no data type is large enough to fit every integer.
    """

    # Find min, max (although built-in min() and max() work for numpy arrays,
    # numpy's implementation is much faster for large numpy arrays)
    if use_numpy:
        data_min = data.min()
        data_max = data.max()

    else:
        data_min = min(data)
        data_max = max(data)

    abs_max = max(abs(data_min), abs(data_max))

    if abs_max <= 127:
        typecode = ('b', 'int8')

    elif abs_max <= 255:

        if data_min >= 0:
            typecode = ('B', 'uint8')
        else:
            typecode = ('h', 'int16')

    elif abs_max <= 32767:
        typecode = ('h', 'int16')

    elif abs_max <= 65535:

        if data_min >= 0:
            typecode = ('H', 'uint16')
        else:
            typecode = ('i', 'int32')

    elif abs_max <= 2147483647:
        typecode = ('i', 'int32')

    elif abs_max <= 4294967295:

        if data_min >= 0:
            typecode = ('I', 'uint32')
        else:
            typecode = ('l', 'int64')

    elif abs_max <= 9223372036854775807:
        typecode = ('l', 'int64')

    elif (abs_max <= 18446744073709551615) and (data_min >= 0):
        typecode = ('L', 'uint64')

    else:
        typecode = (None, None)

    # On Windows (and possibly some uncommon platforms) 'l' and 'L' in ``array.array`` have a max size of
    # 4 bytes instead of expected 8. We check for this, check if an alternate typecode is available,
    # otherwise return None for that typecode.
    if typecode[0] in ('l', 'L'):
        typecode = (_long_long_typecode(typecode[0]), typecode[1])

    return typecode


def data_type_convert_array(data_type, byte_string, use_numpy=False):
    """
    Converts binary data in the form of a byte_string to a flat ``array.array`` or ``np.ndarray`` having
    proper typecode for *data_type*.

    Parameters
    ----------
    data_type : str or unicode
        PDS4 binary data type.
    byte_string : str or bytes
        PDS4 byte string data for an array data structure or a table binary field.
    use_numpy : bool, optional
        If True, returned data will be an ``np.ndarray`` and use NumPy data types. Defaults to False.

    Returns
    -------
    list, array.array or np.ndarray
        Array-like of data converted from a byte string into values having the right data type.

    """

    struct_type = get_binary_struct_type(data_type)
    if struct_type is None:
        raise PDS4StandardsException('Found invalid data type in binary data: {0}'.format(data_type))

    # Extract the byte order and type codes for python and numpy
    byte_order = struct_type[0]
    python_tc = struct_type[1]
    numpy_tc = struct_type[2]

    # On some platforms (e.g. Windows), 8 byte integers are not available on ``array.array``. Thus if we are
    # not using NumPy, we need to direct them into a list. This should only happen for 8 byte integers here
    # since all other PDS4 binary data types should work on all platforms.
    exceeds_array_precision = True if (python_tc is None) else False

    # Unpack byte array data into an array
    if use_numpy:
        byte_string = np.fromstring(byte_string, dtype=numpy_tc)

    elif exceeds_array_precision:

        # Emit a memory efficiency warning
        logger.warning('Detected integer data with precision exceeding memory efficient case. '
                       'Recommend that use_numpy be set to True if data file is large.')

        struct_format = ''

        # Determine byte order
        struct_format += '<' if byte_order == 'little' else '>'

        # Determine number of elements
        struct_format += str(len(byte_string) // 8)

        # Determine format code (struct.unpack, NumPy and array.array typecodes are all not the same)
        if numpy_tc == 'int64':
            struct_format += 'q'
        elif numpy_tc == 'uint64':
            struct_format += 'Q'
        else:
            raise TypeError('Unexpected integer conversion of type: {0}'.format(numpy_tc))

        # Unpack into a list via struct
        compiled_struct = struct.Struct(str(struct_format))
        byte_string = list(compiled_struct.unpack(byte_string))

    else:
        byte_string = arr(str(python_tc), byte_string)

    # Byte swap data if necessary
    need_byteswap = False

    if (byte_order == 'little' and sys.byteorder == 'big') or (byte_order == 'big' and sys.byteorder == 'little'):
        need_byteswap = True

    if need_byteswap:

        if use_numpy:
            byte_string.byteswap(True)
        elif not exceeds_array_precision:
            byte_string.byteswap()

    return byte_string


def data_type_convert_table_ascii(data_type, data, type_aware=False, decode_strings=False, use_numpy=False):
    """
    Converts data originating from a PDS4 Table_Character or Table_Delimited data structure in the form
    of an array_like[byte_string] to a list of values having the proper typecode for *data_type*. Most
    likely this data is a single Field, or a single repetition of a Field, since different Fields have
    different data types.

    Parameters
    ----------
    data_type : str or unicode
        PDS4 data type.
    data : array_like[str or bytes]
        Flat array of PDS4 byte strings from a Table_Character data structure.
    type_aware : bool
        If True, then converts empty or whitespace only values with a numeric data type to 0.
        Defaults to False.
    decode_strings : bool, optional
        If True, strings data types contained in the returned data will be decoded to the ``unicode``
        type in Python 2, and to the ``str`` type in Python 3. If False, leaves string types as byte
        strings. Defaults to False.
    use_numpy : bool, optional
        If True, returned data will be an ``np.ndarray`` and use NumPy data types. Defaults to False.

    Returns
    -------
    list, array.array, or np.ndarray
        Data converted from a byte string array into a values array having the right data type
    """

    # Determine string type for NumPy
    numpy_string_tc = 'U' if decode_strings else 'S'

    # Array struct_type, which is a tuple of (array.array type-code, numpy type-code)
    struct_type = {'ASCII_Real': ('d', 'float64'),
                   'ASCII_Integer': ('l', 'int'),
                   'ASCII_NonNegative_Integer': ('L', 'int'),
                   'ASCII_Boolean': ('b', 'bool_'),
                   'ASCII_Numeric_Base2': ('L', 'int'),
                   'ASCII_Numeric_Base8': ('L', 'int'),
                   'ASCII_Numeric_Base16': ('L', 'int')
                   }.get(data_type, ('c', numpy_string_tc))

    python_tc = struct_type[0]
    numpy_tc = struct_type[1]

    # Special handling for boolean due to e.g. bool('false') = True
    if data_type == 'ASCII_Boolean':

        for i, datum in enumerate(data):

            if datum.strip() in [b'true', b'1']:
                data[i] = True
            else:
                data[i] = False

    # Handle ASCII numeric types and ASCII/UTF-8 strings
    else:

        # Note that we convert binary, octal and hex integers to base 10 integers on the assumption that
        # it is more likely a user will want to do math with them so we cannot store them as strings
        # and to base 10 in order to be consistent on the numerical meaning of all values
        numeric_ascii_type = {'ASCII_Real': ('float', ),
                              'ASCII_Integer': ('int', ),
                              'ASCII_NonNegative_Integer': ('int', ),
                              'ASCII_Numeric_Base2': ('int', 2),
                              'ASCII_Numeric_Base8': ('int', 8),
                              'ASCII_Numeric_Base16': ('int', 16)
                              }.get(data_type, None)

        # Convert ascii numerics into their proper data type
        if numeric_ascii_type is not None:

            # Fill any empty values with a 0, if requested
            if type_aware:

                for i, datum in enumerate(data):
                    if datum.strip() == b'':
                        data[i] = '0'

            # We can use NumPy to convert floats to a numeric type, but not integers. The latter is because
            # in case an integer does not fit into a NumPy C-type (since all ascii integer types are unbounded
            # in PDS4), there appears to be no method to tell NumPy to convert each string to be a numeric
            # Python object. Therefore we use pure Python to convert to numeric Python objects (i.e, int),
            # and then later convert the list into a NumPy array of numeric Python objects.
            if use_numpy and (numpy_tc == 'float64'):

                # Convert to numeric type
                data = np.asarray(data, dtype=numpy_tc)

            else:

                # Convert to numeric type
                cast_func = getattr(builtins, numeric_ascii_type[0])
                args = numeric_ascii_type[1:]

                for i, datum in enumerate(data):
                    data[i] = cast_func(datum, *args)

        # Decode PDS4 ASCII and UTF-8 strings into unicode/str
        elif decode_strings:

            if use_numpy:
                data = np.char.decode(data, 'utf-8')

            else:

                for i, datum in enumerate(data):
                    data[i] = datum.decode('utf-8')

    # Convert to numpy array
    if use_numpy:

        # All ascii integer types are unbounded in PDS4. We cast them to the minimum storage size for memory
        # efficiency, or leave them as python int or long if they are larger than 64-bit
        if (numpy_tc == 'int') and data_type:
            numpy_tc = get_min_integer_struct_type(data)[1]

        data = np.asanyarray(data, dtype=numpy_tc)

    # Convert to array.array
    else:

        # We leave strings in a list
        if python_tc != 'c':

            # All integer types are unbounded in PDS4. We cast them to the minimum storage size for memory
            # efficiency, or leave them as python int or long if they are larger than 64-bit
            if python_tc.lower() == 'l':
                python_tc = get_min_integer_struct_type(data)[0]

            # Convert in-bounds integers and floats, leave everything else as list/python default types
            if python_tc is not None:
                data = arr(str(python_tc), data)

    # Emit memory efficiency warning if necessary
    if (use_numpy and numpy_tc is None) or (not use_numpy and python_tc is None):

        numpy_suggest = ''
        if not use_numpy:
            numpy_suggest = ' Recommend that use_numpy be set to True if data file is large.'

        logger.warning('Detected integer Field with precision exceeding memory efficient case.'
                        + numpy_suggest)

    return data


def data_type_convert_table_binary(data_type, data, decode_strings=False, use_numpy=False):
    """
    Converts data originating from a PDS4 Table_Binary data structure in the form of an
    array_like[byte_string] to a list of values having the proper typecode for *data_type*. Most likely
    this data is a single Field, or a single repetition of a Field, since different Fields have different
    data types.

    Parameters
    ----------
    data_type : str or unicode
        PDS4 data type.
    data : array_like[str or bytes]
        Flat array of PDS4 byte strings from a Table_Binary data structure.
    decode_strings : bool, optional
        If True, strings data types contained in the returned data will be decoded to the ``unicode``
        type in Python 2, and to the ``str`` type in Python 3. If False, leaves string types as byte
        strings. Defaults to False.
    use_numpy : bool, optional
        If True, returned data will be an ``np.ndarray`` and use NumPy data types. Defaults to False.

    Returns
    -------
    list, array.array, or np.ndarray
        Data converted from a byte string array into a values array having the right data type.
    """

    binary_type = get_binary_struct_type(data_type)

    # Convert character data types
    if binary_type is None:
        data = data_type_convert_table_ascii(data_type, data,
                                             decode_strings=decode_strings, use_numpy=use_numpy)

    # Convert binary data types
    else:

        # Join data list back into a byte_string
        byte_string = b''.join(data)

        data = data_type_convert_array(data_type, byte_string, use_numpy=use_numpy)

    return data


def adjust_array_data_type(array, scaling_factor, value_offset, use_numpy=False):
    """
    Converts the input *array*, in-place if possible, into a new large enough data type if adjusting said
    array as-is by *scaling_factor* or *value_offset* would result in an overflow. This can be necessary both
    if the array is data from a PDS4 Array or a PDS4 Table, so long as it has a scaling factor or value
    offset associated with it.

    Parameters
    ----------
    array : array_like
        Any PDS4 numeric data.
    scaling_factor : int or float
        PDS4 scaling factor to apply to data.
    value_offset : int or float
        PDS4 value offset to apply to data.
    use_numpy : bool, optional
        If True, assumed that input *array* is an ``np.ndarray`` and output will also be an ``np.ndarray``.
        Defaults to False.

    Returns
    -------
    list, array.array, or np.ndarray
        Original *array* modified to have a new data type.
    """

    # This method is unnecessary for lists containing python's default numeric types because python
    # will automatically perform the type conversion if needed
    if isinstance(array, (list, tuple)):
        return array

    scaling_is_float = isinstance(scaling_factor, float)
    offset_is_float = isinstance(value_offset, float)

    if use_numpy:
        data_is_float = np.issubdtype(array.dtype, float)
        data_is_double = array.dtype == np.float64
    else:
        data_is_float = array.typecode in ('f', 'd')
        data_is_double = array.typecode == 'd'

    # Set to double
    if data_is_float or scaling_is_float or offset_is_float:

        # For floats, we always convert to 64-bit, even if 32-bit could have fit
        if not data_is_double:

            if use_numpy:

                try:
                    array = array.astype('float64', copy=False)
                except TypeError:
                    array = array.astype('float64')

            else:
                array = arr(str('d'), array)

    # Set to int
    else:

        # For ints, we find minimum size necessary for data not to overflow
        if scaling_factor is None:
            scaling_factor = 1

        if value_offset is None:
            value_offset = 0

        # For numpy, adjust to new data type if necessary
        if use_numpy:

            min_data = int(array.min()) * scaling_factor + value_offset
            max_data = int(array.max()) * scaling_factor + value_offset

            new_typecode = get_min_integer_struct_type([min_data, max_data])[1]

            if new_typecode is None:
                new_typecode = np.asarray([min_data, max_data]).dtype

                logger.warning('Detected integer Field with precision exceeding memory efficient case.')
            else:
                new_typecode = np.dtype(new_typecode)

            has_new_type = str(new_typecode) != str(array.dtype)
            no_precision_loss = (new_typecode.itemsize >= array.dtype.itemsize) or (str(new_typecode) == 'object')

            if has_new_type and no_precision_loss:

                try:
                    array = array.astype(new_typecode, copy=False)
                except TypeError:
                    array = array.astype(new_typecode)

        # For array.array, adjust to new data type if necessary
        else:

            min_data = int(min(array)) * scaling_factor + value_offset
            max_data = int(max(array)) * scaling_factor + value_offset

            new_typecode = get_min_integer_struct_type([min_data, max_data])[0]

            if new_typecode is None:
                array = list(array)

                logger.warning('Detected integer Field with precision exceeding memory efficient case. '
                               'Recommend that use_numpy be set to True if data file is large.')

            else:

                has_new_type = new_typecode != array.typecode
                no_precision_loss = arr(str(new_typecode)).itemsize >= array.itemsize

                if has_new_type and no_precision_loss:
                    array = arr(str(new_typecode), array)

    return array


def adjust_complex_type(array, use_numpy=False):
    """
    Assumes an array with length evenly divisible by two is passed in, where each two sequential values are
    the real and imaginary parts of a single complex value. Creates a complex() type if use_numpy is off,
    and a NumPy complex type otherwise.

    Parameters
    ----------
    array : array_like
        Any PDS4 numeric data.
    use_numpy : bool, optional
        If True, assumed that input *array* is an ``np.ndarray`` and output will also be an ``np.ndarray``,
        and use NumPy's complex type. Defaults to False.

    Returns
    -------
    list or np.ndarray
        Same data as original *array* but with a new data type.

    """

    if use_numpy:
        new_array_like = array[::2] + 1j * array[1::2]

    else:
        new_array_like = [complex(array[i], array[i + 1]) for i in range(0, len(array), 2)]

    return new_array_like


def apply_scaling_and_value_offset(data, scaling_factor, value_offset, use_numpy=False):
    """ Applies scaling factor and value offset to *data*.

    Data is modified in-place, if possible. Data type may change to prevent numerical overflow
    if applying scaling factor and value offset would cause one.

    Parameters
    ----------
    data : array_like
        Any numeric PDS4 data.
    scaling_factor : int or float
        PDS4 scaling factor to apply to data.
    value_offset : int or float
        PDS4 value offset to apply to data.
    use_numpy : bool, optional
        If True, assumes passed in array is some form of a NumPy array. Defaults to False.

    Returns
    -------
    array_like
        *data* with *scaling_factor* and *value_offset* applied. May have different data type.
    """

    # Skip taking computationally intensive action if no adjustment is necessary
    if scaling_factor == 1 and value_offset == 0:
        return data

    # Adjust data type to prevent overflow on application of scaling factor and value offset, if necessary
    data = adjust_array_data_type(data, scaling_factor, value_offset, use_numpy)

    # Apply scaling factor and value offset
    if use_numpy:
        if scaling_factor is not None:
            data *= scaling_factor

        if value_offset is not None:
            data += value_offset

    else:
        for i, datum in enumerate(data):

            if scaling_factor is not None:
                data[i] *= scaling_factor

            if value_offset is not None:
                data[i] += value_offset

    return data


def _long_long_typecode(python_typecode):
    """

    Python 2, and Python 3 prior to v3.3 do not have an 8 byte integer data type for ``array.array`` on
    Windows due to the underlying C compiler, this is also possibly true on other rarer platforms
    (tested Linux and Mac flavors did have it). The usual 'l' and 'L' typecodes are 4 bytes on these systems,
    rather than 8 bytes. Python 3.3 introduces the 'q' and 'Q' typecodes that are 8 byte integers on
    any platform on which they are supported.

    The purpose of this method is to adjust the input long-long typecode to a long-long typecode supported
    by this platform, or None if no such typecode exists.

    Notes
    -----
    The reasoning in this method applies to ``array.array``. The ``struct`` method functions differently
    and generally supports 8 byte integers no matter what Python version.

    Parameters
    ----------
    python_typecode : str or unicode
        An ``array.array`` typecode that could support 8 byte integers on certain platforms.
        One of  l, L, q, Q.

    Returns
    -------
    str or None
        Returns an ``array.array`` typecode that will support 8 byte integers on this platform, or None if
        this platform does not support 8 byte integers in ``array.array``.

    """

    # Raise an error if a non-long-long typecode passed
    if python_typecode not in ('l', 'L', 'q', 'Q'):
        raise ValueError("Input typecode, '{0}', must be a valid 8 byte integer typecode (l, L, q, Q)."
                         .format(python_typecode))

    long_long_is_32bit = arr(str('l')).itemsize == 4 or arr(str('L')).itemsize == 4

    # If a long-long type is passed and the standard long-long is 4 bytes as opposed to the expected 8 bytes
    if long_long_is_32bit:

        # Check if specific 8 byte integer typecode introduced in Python 3.3+ is available
        try:
            arr(str('q'))

        # Return that no long long typecode exists if it is not
        except ValueError:
            long_long_typecode = None

        # Determine the proper long long typecode to correspond to the initially passed in typecode
        else:

            if python_typecode.isupper():
                long_long_typecode = 'Q'

            else:
                long_long_typecode = 'q'

    # If the standard long-long is already 8 bytes then we pass it back as the long-long of choice
    else:

        long_long_typecode = python_typecode

    # Typecast to str as necessary; under Python 2, ``array.array`` cannot take unicode
    if long_long_typecode is not None:
        long_long_typecode = str(long_long_typecode)

    return long_long_typecode
