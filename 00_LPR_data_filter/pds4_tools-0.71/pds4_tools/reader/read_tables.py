from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
from functools import reduce
from math import log10

from .read_arrays import apply_scaling_and_value_offset, flat_data_to_list
from .table_objects import (TableStructure, TableManifest, Meta_FieldUniformlySampled)
from .data_types import (data_type_convert_table_ascii, data_type_convert_table_binary,
                         get_min_integer_struct_type, adjust_complex_type)

from ..utils.constants import PDS4_TABLE_TYPES
from ..utils.logging import logger_init
from ..utils.exceptions import PDS4StandardsException

from ..extern import six

# Safe import of numpy (not required)
try:
    import numpy as np
except ImportError:
    np = None

# Initialize the logger
logger = logger_init()


#################################


def _read_table_byte_data(table_structure):
    """ Reads the byte data from the data file for a PDS4 Table.

    Determines, from the structure's meta data, the relevant start and stop bytes in the data file prior to
    reading. For fixed-width tables (Table_Character and Table_Binary), the returned data is exact. For
    Table_Delimited, the byte data is likely to go beyond end of its last record.

    Parameters
    ----------
    table_structure : TableStructure
        The PDS4 Table data structure for which the byte data needs to be read. Should have been
        initialized via `TableStructure.fromfile` method, or contain the required meta data.

    Returns
    -------
    str or bytes
        The byte data for the table.
    """

    from .core import read_byte_data

    meta_data = table_structure.meta_data
    num_records = meta_data['records']
    start_byte = meta_data['offset']

    if meta_data.is_fixed_width():

        record_length = meta_data.record['record_length']
        stop_byte = start_byte + num_records * record_length

    elif meta_data.is_delimited():

        object_length = meta_data.get('object_length')
        record_length = meta_data.record.get('maximum_record_length')

        if object_length is not None:
            stop_byte = start_byte + object_length

        elif record_length is not None:
            stop_byte = start_byte * num_records * record_length

        else:
            stop_byte = -1

    else:
        raise TypeError('Unknown table type: {0}'.format(table_structure.type))

    return read_byte_data(table_structure.parent_filename, start_byte, stop_byte)


def table_data_size_check(table_structure, quiet=False):
    """ Checks, and warns, if table is estimated to have a large amount of data.

    This estimate is done from the meta-data only and excludes nested fields (fields inside groups fields)
    and repetitions. A more accurate meta-data only estimate could be obtained via `TableManifest`.

    Parameters
    ----------
    table_structure : Structure
        The table structure whose data to check for size.
    quiet : bool, optional
        If True, does not output warning if table contains a large amount of data. Defaults to False.

    Returns
    -------
    bool
        True if the table structure exceeds pre-defined parameters for size of its data, False otherwise.
    """

    meta_data = table_structure.meta_data
    dimensions = meta_data.dimensions()

    # Estimate of the number of elements in the table
    num_elements = dimensions[0] * dimensions[1]

    # Limit at which the data is considered large
    if meta_data.is_delimited():

        # Loading delimited tables is slower than fixed-width tables due to additional required processing
        num_elements_warn = 2 * 10**7

    else:
        num_elements_warn = 4 * 10**7

    if num_elements > num_elements_warn:

        if not quiet:
            logger.info("{0} contains a large amount of data. Loading data may take a while..."
                        .format(table_structure.id))

        return True

    return False


def _make_uniformly_sampled_field(table_structure, use_numpy=False):
    """ Add data for the Uniformly_Sampled field to *table_structure*.

    Creates and populates a Uniformly_Sampled field based on PDS4 label description.
    Modifies table_structure to add newly created field.

    Parameters
    ----------
    table_structure : TableStructure
        The PDS4 Table data structure which should contain a Uniformly_Sampled field.
    use_numpy : bool, optional
        If True, added data will be an ``np.ndarray`` and use NumPy data types. Defaults to False.

    Returns
    -------
    None
    """

    uniformly_sampled_xml = table_structure.label.find('Uniformly_Sampled')
    uniformly_sampled_data = []

    # Do nothing if there is no Uniformly Sampled field in this table
    if uniformly_sampled_xml is None:
        return

    # Read the meta-data for the Uniformly_Sampled field
    try:

        uni_sampled = Meta_FieldUniformlySampled()
        uni_sampled.load(uniformly_sampled_xml)

    except PDS4StandardsException as e:
        logger.warning('Unable to create Uniformly Sampled field: ' + six.text_type(e))
        return

    # Extract scale (older PDS4 standards allowed leaving it empty)
    if 'scale' not in uni_sampled:
        scale = 'linear'

    else:
        scale = uni_sampled['scale'].lower()

    # Extract necessary values to speed up calculation
    num_records = table_structure.meta_data['records']
    last_value = uni_sampled['last_value']
    interval = uni_sampled['interval']

    # Calculate field's data for Linear sampling
    if scale == 'linear':

        current_value = uni_sampled['first_value']

        for j in range(0, num_records):

            uniformly_sampled_data.append(current_value)
            current_value += interval

    # Calculate field's data for Logarithmic sampling
    elif scale == 'logarithmic':

        # Implements xj = x1 * (xn/x1)^[(j-1)/(n-1)] for j = 1 ... n, where xn/x1^(1/(n-1)) is the
        # interval and x1 ... xj are the field's data.

        x1 = uni_sampled['first_value']

        for j in range(0, num_records):

            current_value = x1 * (interval ** j)
            uniformly_sampled_data.append(current_value)

    # Calculate field's data for Exponential sampling
    elif scale == 'exponential':

        # Implements b^xj = b^x1 + (j-1)*(b^xn - b^x1)/(n-1) for j = 1 ... n, where (b^xn - b^x1)/(n-1)
        # is the interval and x1 ... xj are the field's data.

        base = uni_sampled['base']

        log_x1 = base ** uni_sampled['first_value']
        log_base = log10(base)

        for j in range(0, num_records):

            current_value = log10(log_x1 + j * interval) / log_base
            uniformly_sampled_data.append(current_value)

    # Function to compare closeness of two floating point numbers, based on PEP-0485 with larger tolerance
    def is_close_num(a, b, rel_tol=1e-3, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    # Warn if last calculated value for Uniformly Sampled does not match indicated last value in label
    if not is_close_num(last_value, uniformly_sampled_data[-1]):
        logger.warning("Last value in Uniformly Sampled field, '{0}', does not match expected '{1}'."
                       .format(uniformly_sampled_data[-1], last_value))

    table_structure.add_field(uniformly_sampled_data, uni_sampled, use_numpy)


def _extract_fixed_width_field_data(extracted_data, table_byte_data,
                                    field_length, field_location, record_length,
                                    array_shape, group_locations=None, repetition_lengths=None,
                                    _dimension=0, _current_position=None):
    """
    Extracts data for a single field in a fixed-width (Character or Binary) table.

    Parameters
    ----------
    extracted_data : list[str or bytes]
        The extracted byte data for each element the table. Should be an empty list on initial call.
        Modified in-place.
    table_byte_data : str or bytes
        Byte data for the entire table.
    field_length : int
        Length of each element in the field, in bytes.
    field_location : int
        Location of the first element in the field, in bytes, from the beginning of the table.
    record_length : int
        Length of each record in the table, in bytes.
    array_shape : array_like[int]
        Sequence of dimensions for the field. First element is the number of records, all other
        elements are the number of repetitions for each GROUP the field is inside of, if any.
    group_locations : array_like[int], optional
        If this field is inside at least one group field, the list must contain the location of the
        first element of the first repetition (i.e, group_location), in bytes, of each group.
    repetition_lengths : array_like[int], optional
        If this field is inside at least one group field, the list must contain the group length divided
        by the number of repetitions (i.e, group_length/repetitions), in bytes, for each group.
    _dimension : int, optional
        The current dimension of `array_shape` being looped over. Should be 0 on initial call.
    _current_position : array_like[int], optional
        The position inside `array_shape` being looped over. Should be None on initial call.

    Returns
    -------
    None
    """

    # Simplified, sped up, case for fields that are not inside group fields
    if len(array_shape) == 1:

        # Loop over each record, extracting the datum for this field
        for record_num in range(0, array_shape[0]):

            start_byte = (record_num * record_length) + field_location
            stop_byte = start_byte + field_length

            extracted_data.append(table_byte_data[start_byte:stop_byte])

        return

    # On initial call (prior to recursion) for fields with groups
    if _dimension == 0:

        # Determine if the data for a field inside group fields is contiguous. If it is, we can significantly
        # speed operations up. To determine if contiguous, we check that all group locations (except possibly
        # the first) start from the first byte, and that the group_length/group_repetitions of each group is
        # equal to the group_length of its child group. Finally, we check that the group_length/group_repetitions
        # for the last group is equal to the field length.
        has_contiguous_locations = group_locations[1:] == [0] * (len(group_locations) - 1)
        has_contiguous_end_bytes = field_length == repetition_lengths[-1]

        for i, length in enumerate(repetition_lengths[1:]):

            if length * array_shape[i+2] != repetition_lengths[i]:
                has_contiguous_end_bytes = False

        # Simplified, sped up, case for fields inside group fields that are contiguous. Principle of
        # operation is that we can effectively read one element after the other, as we would in an array,
        # where the only thing we need to know is the element length. A slight complication is that we
        # need to take into account the byte jump between end of the field for one record and start of it
        # for the next record.
        if has_contiguous_locations and has_contiguous_end_bytes:

            group_start_byte = group_locations[0]
            num_elements = reduce(lambda x, y: x*y, array_shape[1:])

            # Loop over each record
            for record_num in range(0, array_shape[0]):

                record_start_byte = (record_num * record_length) + group_start_byte + field_location

                # Loop over each element (multiple due to group field) in the record for this field,
                # extracting them
                for element_num in range(0, num_elements):
                    start_byte = record_start_byte + field_length * element_num
                    stop_byte = start_byte + field_length

                    extracted_data.append(table_byte_data[start_byte:stop_byte])

            return

        # If we've reached this point then our field is inside group fields, and those group fields are not
        # contiguous. Therefore we will need to calculate each and every position. Here we do the initial
        # setup prior to launching into recursion described below.
        _current_position = [0] * len(array_shape)
        group_locations = group_locations if group_locations else []
        repetition_lengths = repetition_lengths if repetition_lengths else []

    # Extract each element's byte data if we have its full position in array_shape, via the formula:
    # start_byte = record_length * i_current_record_number
    # + first_group_location + (first_group_length/first_group_repetitions) * j_current_first_group_repetition
    # + (repeat) n_group_location + (n_group_length/n_group_repetitions) * k_current_n_group_repetition
    # + field_location
    # stop_byte = start_byte + field_length
    if _dimension == len(array_shape):

        start_byte = record_length * _current_position[0] + field_location

        for i in range(0, len(group_locations)):
            start_byte += group_locations[i] + repetition_lengths[i] * _current_position[i + 1]

        stop_byte = start_byte + field_length
        extracted_data.append(table_byte_data[start_byte:stop_byte])

    # Create nested for-loops (recursive) over each dimension in `array_shape`, such that `current_positions`
    # will contain every possible valid combination of the dimension values in `array_shape`. E.g., if the
    # shape of the field (due to a GROUP) is [100, 50], then the `current_positions` created by these nested
    # for loops will have positions [0, 0], [0, 1], ... [0, 49], [1, 0], ... [1, 49], [2, 0] ... [99, 49].
    # The deepest loop therefore has a `current_position` that contains the record number and the group
    # repetition numbers for each GROUP the field is in. In case of field that is not inside any groups, this
    # simplifies to a single for-loop which loops over each record in the field.
    else:

        for i in range(0, array_shape[_dimension]):

            _extract_fixed_width_field_data(extracted_data, table_byte_data,
                                            field_length, field_location, record_length,
                                            array_shape, group_locations, repetition_lengths,
                                            _dimension + 1, _current_position)

            _current_position[_dimension] += 1

        _current_position[_dimension] = 0


def _extract_delimited_field_data(extracted_data, table_byte_data,
                                  start_bytes, current_column,
                                  array_shape):
    """
    Extracts data for a single field in a delimited table.

    Parameters
    ----------
    extracted_data : list[str or bytes]
        The extracted byte data for each element the table. Should be an empty list on initial call.
        Modified in-place.
    table_byte_data : array_like[str or bytes]
        Byte data for the entire table, split into records.
    start_bytes : array_like
        The start byte for each element in the table. Two-dimensional, where the first dimension specifies
        which column (if the record were split by delimiter) and the second dimension specifies which record.
    current_column : int
        Specifies which column (if the record were split by delimiter) to extract the data for. For fields
        that are inside GROUPs, this is the first column and columns up until the number of repetitions
        will be extracted. For tables without GROUP fields, this is equivalent to the field number.
    array_shape array_like[int]
        Sequence of dimensions for the field. First element is the number of records, all other
        elements are the number of repetitions for each GROUP the field is inside of, if any.

    Returns
    -------
    None
    """

    num_records = array_shape[0]
    num_group_columns = 0 if (len(array_shape) == 1) else reduce(lambda x, y: x*y, array_shape[1:]) - 1

    # Extract data from GROUP fields. This can also be used for non-GROUP fields but because of the
    # setup time for the inner loop (which would only loop once) it is approximately 25% slower than
    # the non-GROUP version below.
    if num_group_columns > 0:

        # Pre-extract necessary variables to speed up computation time
        last_column = current_column + num_group_columns + 1

        # Extract the byte data for this specific column from byte_data
        for record_num in range(0, num_records):

            for column_num in range(current_column, last_column):

                start_idx = start_bytes[column_num][record_num]
                end_idx = start_bytes[column_num + 1][record_num] - 1

                extracted_data.append(table_byte_data[record_num][start_idx:end_idx])

        # Remove start_bytes for no-longer needed columns to save memory
        start_bytes[current_column:last_column] = [None] * (num_group_columns + 1)

    # Extract data from non-GROUP fields
    else:

        # Pre-extract necessary variables to speed up computation time
        field_start_bytes = start_bytes[current_column]
        next_field_start_bytes = start_bytes[current_column + 1]

        # Extract the byte data for this specific column from byte_data
        for record_num in range(0, num_records):
            start_idx = field_start_bytes[record_num]
            end_idx = next_field_start_bytes[record_num] - 1

            extracted_data.append(table_byte_data[record_num][start_idx:end_idx])

        # Remove start_bytes for no-longer needed columns to save memory
        start_bytes[current_column] = None


def _get_delimited_records_and_start_bytes(records, table_structure, table_manifest):
    """
    For a delimited table, we obtain the start byte of each field (and each repetition of field)
    for each record, and adjust the records themselves such that any field value starts at its start byte
    and ends at the start byte of the next field value.

    In principle there are a number of ways to read a delimited table. One could read it in row by row:
    however when converting each value to a data type, there is CPU overhead in determining what that data
    type should be and the conversion itself. If we read a row, and then convert each value for that row one
    at a time then we have that overhead each time and that becomes extremely costly for large numbers of
    records. If we could read an entire column at a time and convert it then we avoid said overhead. One
    easy approach to the latter is to store a 2D array_like[record, field], thus splitting the data first
    into records and then each record by delimiter (adjusting to account for double quote where needed).
    However in general this approach is often very memory intensive because the strings, especially with
    overhead, require more memory to store than once the data is converted to its desired type. Instead, we
    take a similar approach where we use a 2D array_like[field, record] to record the start bytes of the data
    for each field but do not actually split each record into fields. We try to use the memory efficient
    ``array.array`` and minimal size required to store each start byte, this will nearly always result in
    significantly less memory than would be required to actually split each field into a string since the
    start bytes will nearly always be 1 or 2 bytes each because records are rarely longer than 65535 characters.

    Parameters
    ----------
    records : list[str or bytes]
        The data for the delimited table, split into records.
    table_structure : TableStructure
        The PDS4 Table data structure for the delimited table.
    table_manifest : TableDelimitedManifest
        A manifest describing the structure of the PDS4 delimited table.

    Returns
    -------
    list[str or bytes], list[array.array or list]
        A two-valued tuple of: the records for the table, modified to remove quotes; and a 2D array_like,
        where the first dimension is the field and the second dimension is the record, with the value being
        the start byte of the data in the first tuple value for those parameters.
    """

    # Extract the proper record delimiter (as bytes, for compatibility with Python 3)
    delimiter_name = table_structure.meta_data['field_delimiter'].lower()
    field_delimiter = {'comma': b',',
                       'horizontal tab': b'\t',
                       'semicolon': b':',
                       'vertical bar': b'|'
                      }.get(delimiter_name, None)

    # Determine total number of columns (if we split the record by record delimiter) in each record.
    # A column is either a field or if there's a GROUP then it's one of the repetitions of a field.
    # This number, as are other references to this number in the code, are corrected for cases where
    # we ignore the record delimiter as its effectively escaped by being between bounding double quotes.
    num_columns = 0

    for field in table_manifest.fields():

        repetitions = []
        parent_idx = table_manifest.index(field)

        for j in range(0, field.group_level):
            parent_idx = table_manifest.get_parent_by_idx(parent_idx, return_idx=True)
            repetitions.append(table_manifest[parent_idx]['repetitions'])

        num_columns += 1 if (not repetitions) else reduce(lambda x, y: x*y, repetitions)

    # Pre-allocate ``list``, which will store either ``array.array``s or other ``list``s that contain the
    # start byte of each field for each record. Thus `start_bytes` is a two-dimensional array_like, where
    # the first dimension is the field and the second dimension is the record, with the value being the
    # start byte of the data for those parameters.
    start_bytes = [None] * (num_columns + 1)

    longest_record = len(max(records, key=len))
    array_type = get_min_integer_struct_type([longest_record + 1])[0]

    for i in range(0, num_columns + 1):
        if array_type is None:
            start_bytes[i] = [None] * len(records)
        else:
            start_bytes[i] = array.array(str(array_type), [0] * len(records))

    # In Python 3, when checking for the first and last character below, we need to convert the value
    # to a ``str``, since obtaining first character of a ``bytes`` returns a byte value. In Python 2, no
    # action needs to be taken since the value is ``str`` by default.
    if six.PY2:
        str_args = ()
    else:
        str_args = ('utf-8', )

    # Obtain start bytes for each field in each record, and adjust the record itself such that only the
    # start byte is required to obtain the entire value (to save RAM). The latter is needed due to the
    # requirement to ignore delimiters found inside enclosing quotes and that such enclosing quotes themselves
    # are not part of the value: i.e., a record consisting of '"value1", value2' would want to use the start
    # byte of 'value2' as the end byte of '"value1"', but this would include the extra quote at the end,
    # therefore we remove such quotes after recording the proper start byte.
    for record_idx, record in enumerate(records):

        # Split the record by delimiter
        split_record = record.split(field_delimiter)
        next_start_byte = 0

        # Look for field values bounded by a double quotes. Inside such values any delimiter found should be
        # ignored, but ``split`` above will not ignore it. Therefore we have to join the value back.
        if b'"' in record:

            split_record_len = len(split_record)
            field_idx = 0

            # Loop over each field value (may turn out to only be part of a field)
            while field_idx < split_record_len:
                value = split_record[field_idx]
                value_length = len(value)
                first_character = str(value, *str_args)[0] if value_length > 0 else None

                # If field value starts with a quote then we need to check if there is a matching closing
                # quote somewhere further.
                if first_character == '"':

                    next_quote_idx = -1

                    # Find the index of the field value containing the next quote in the record
                    if b'"' in value[1:]:
                        next_quote_idx = field_idx

                    else:

                        for k, next_value in enumerate(split_record[field_idx + 1:]):
                            if b'"' in next_value:
                                next_quote_idx = field_idx + 1 + k
                                break

                    # If a latter or same field value contained a quote, check whether it was the last
                    # character in the value (and thus the two quotes enclosed a single value)
                    if next_quote_idx >= 0:
                        next_value = split_record[next_quote_idx]
                        last_character = str(next_value, *str_args)[-1] if len(next_value) > 0 else None

                        if last_character == '"':

                            # Reconstruct the original value prior to ``split``
                            original_value = field_delimiter.join(split_record[field_idx:next_quote_idx+1])

                            # Remove the quote at the start and end of the original value
                            original_value = original_value[1:-1]

                            # Insert the joined value back into split_record (and remove its split components)
                            split_record = split_record[0:field_idx] + [original_value] + \
                                           split_record[next_quote_idx+1:]

                            # We've joined several values into one, therefore split_record_len has shrunk
                            split_record_len -= next_quote_idx - field_idx

                            # Record the start byte of this field, and adjust the next start byte to account
                            # for the entire length of the joined field
                            start_bytes[field_idx][record_idx] = next_start_byte
                            next_start_byte += len(original_value) + 1

                            field_idx += 1
                            continue

                # If the record had a quote somewhere but not in this field or it was not an enclosing quote
                # for the field then we simply record its start byte and set the next start byte as usual
                start_bytes[field_idx][record_idx] = next_start_byte
                next_start_byte += value_length + 1

                field_idx += 1

            # Join (the potentially) adjusted record back into a single string to save ``str`` overhead memory
            records[record_idx] = field_delimiter.join(split_record)

        # If there were no quotes in the record then we can simply record the start bytes of each value
        # (surprisingly splitting the record and doing this via length of each value appears to be the
        # fastest way to accomplish this since ``str.split`` is written in C.)
        else:

            for field_idx, value in enumerate(split_record):
                start_bytes[field_idx][record_idx] = next_start_byte
                next_start_byte += len(value) + 1

        # Add an extra start byte, which actually acts only as the end byte for the last field
        start_bytes[-1][record_idx] = next_start_byte

    return records, start_bytes


def read_table_data(table_structure, no_scale, decode_strings, use_numpy):
    """
    Reads and properly formats the data for a single PDS4 table structure, modifies *table_structure* to
    contain all extracted fields for said table.

    Parameters
    ----------
    table_structure : TableStructure
        The PDS4 Table data structure to which the table's data fields should be added.  Should have been
        initialized via `TableStructure.fromfile` method.
    no_scale : bool
        Returned data will not be adjusted according to the offset and scaling factor. Defaults to False.
    decode_strings : bool
        If True, strings data types contained in the returned data will be decoded to the ``unicode`` type
        in Python 2, and to the ``str`` type in Python 3. If False, leaves string types as byte strings.
    use_numpy : bool
        If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.
    Returns
    -------
    None
    """

    # Provide a warning to the user if the data is large and may take a while to read
    table_data_size_check(table_structure)

    # Obtain the byte data of the table
    table_byte_data = _read_table_byte_data(table_structure)

    # Obtain a manifest for the table, which describes the table structure (the fields and groups)
    table_manifest = TableManifest(table_structure.label)

    # Add Uniformly_Sampled field to the table if it exists
    _make_uniformly_sampled_field(table_structure, use_numpy)

    # Extract the number of records
    num_records = table_structure.meta_data['records']

    # Special processing for delimited tables
    if table_structure.meta_data.is_delimited():

        # Split the byte data into records
        table_byte_data = table_byte_data.split(b'\r\n')[0:num_records]

        # Obtain adjusted records (to remove quotes) and start bytes (2D array_like, with first dimension
        # the field number and the second dimension the record number, and the value set to the start byte
        # of the data for those parameters).
        table_byte_data, start_bytes = _get_delimited_records_and_start_bytes(table_byte_data,
                                                                              table_structure, table_manifest)

        # For delimited data, we can split each record by the delimiter. In the loop over fields below,
        # this number represents which column of `start_bytes` has the data for the field being looped over.
        # In tables without GROUP fields, this is equivalent to the field number.
        current_column = 0

    # This for loop creates the majority of 'data' variable, which will store all columns and their values
    # for the table whose manifest was obtained. It inserts each field (and each repetition, for group fields)
    # one at a time
    for field in table_manifest.fields():

        # Stores the shape of the array-like which will contain the data for this field
        # (modified in for loop below to add dimensions for group repetitions)
        array_shape = [num_records]

        # Stores the group_location of each group the field is inside of (added in for loop below)
        group_locations = []

        # Stores the group_location divided by the repetitions for each group the field is inside of
        # (added in for loop below)
        repetition_lengths = []

        parent_idx = table_manifest.index(field)
        for j in range(0, field.group_level):
            parent_idx = table_manifest.get_parent_by_idx(parent_idx, return_idx=True)
            parent_group = table_manifest[parent_idx]

            array_shape.insert(1, parent_group['repetitions'])

            if table_structure.meta_data.is_fixed_width():

                group_locations.insert(0, parent_group['location'] - 1)
                repetition_lengths.insert(0, parent_group['length'] // parent_group['repetitions'])

        # Create flat list that will contain the (flat) data for this Field
        extracted_data = []

        # Extract the byte data for the field (delimited tables)
        if table_structure.meta_data.is_delimited():

            # Determine number of repetitions there are (each of these is effectively a column in the record)
            num_group_columns = 0
            if len(array_shape) > 1:
                num_group_columns = reduce(lambda x, y: x*y, array_shape[1:]) - 1

            _extract_delimited_field_data(extracted_data, table_byte_data,
                                          start_bytes, current_column, array_shape)

            current_column += 1 + num_group_columns

        # Extract the byte data for the field (fixed-width tables)
        else:

            record_length = table_structure.meta_data.record['record_length']

            _extract_fixed_width_field_data(extracted_data, table_byte_data, field['length'],
                                            field['location'] - 1, record_length,
                                            array_shape, group_locations, repetition_lengths)

        # Cast the byte data for this field into the appropriate data type
        try:
            kwargs = {'decode_strings': decode_strings, 'use_numpy': use_numpy}

            if table_structure.type == 'Table_Character':
                extracted_data = data_type_convert_table_ascii(field['data_type'], extracted_data, **kwargs)

            elif table_structure.type == 'Table_Binary':
                extracted_data = data_type_convert_table_binary(field['data_type'], extracted_data, **kwargs)

            elif table_structure.meta_data.is_delimited():
                extracted_data = data_type_convert_table_ascii(field['data_type'], extracted_data,
                                                               type_aware=True, **kwargs)

            else:
                raise TypeError('Unknown table type: {0}'.format(table_structure.type))

        except ValueError as e:
            six.raise_from(ValueError("Unable to convert field '{0}' to data_type '{1}': {2}"
                                      .format(field['name'], field['data_type'], e)), None)

        # Make scaling_factor and value_offset adjustments
        scaling_factor = field.get('scaling_factor')
        value_offset = field.get('value_offset')

        # Adjust data values to account for 'scaling_factor' and 'value_offset' (in-place if possible)
        # (Note that this may change the data type to prevent overflow and thus increase memory usage)
        if (not no_scale) and ((scaling_factor is not None) or (value_offset is not None)):
            extracted_data = apply_scaling_and_value_offset(extracted_data, scaling_factor,
                                                            value_offset, use_numpy)

        # Adjust data to use complex data types if it had them
        if 'complex' in field['data_type'].lower():
            extracted_data = adjust_complex_type(extracted_data, use_numpy)

        # For fields inside groups, we extract data from its current flat (sequential) structure
        # into a properly formatted array (simulated via nested lists)
        if len(array_shape) > 1:
            extracted_data = flat_data_to_list(extracted_data, array_shape)

        # Add read-in field to the TableStructure
        table_structure.add_field(extracted_data, field, use_numpy)


def read_table(full_label, table_label, data_filename,
               lazy_load=False, no_scale=False, decode_strings=False, use_numpy=False):
    """ Create the `TableStructure`, containing label, data and meta data for a PDS4 Table.

    Used for all forms of PDS4 Tables (i.e., Table_Character, Table_Binary and Table_Delimited).

    Parameters
    ----------
    full_label : Label
        The entire label for a PDS4 product, from which *table_label* originated.
    table_label : Label
        Portion of label that defines the PDS4 table data structure.
    data_filename : str or unicode
        Filename, including the full path, of the data file that contains the data for this table.
    lazy_load : bool, optional
        If True, does not read-in the data of this table until the first attempt to access it.
        Defaults to False.
    no_scale : bool, optional
        If True, returned data will not be adjusted according to the offset and scaling
        factor. Defaults to False.
    decode_strings : bool, optional
        If True, strings data types contained in the returned data will be decoded to
        the ``unicode`` type in Python 2, and to the ``str`` type in Python 3. If
        false, leaves string types as byte strings. Defaults to False.
    use_numpy : bool, optional
        If True, extracted data will use ``np.ndarray``'s and NumPy data types. Defaults to False.

    Returns
    -------
    TableStructure
        An object representing the table; contains its label, data and meta data.

    Raises
    ------
    TypeError
        Raised if called on a non-table according to *table_label*.
    """

    # Skip over data structure if its not actually a supported Table
    if table_label.tag not in PDS4_TABLE_TYPES:
        raise TypeError('Attempted to read_table() on a non-table: ' + table_label.tag)

    # Create the data structure for this table
    table_structure = TableStructure.fromfile(data_filename, table_label, full_label,
                                              lazy_load=lazy_load, no_scale=no_scale,
                                              decode_strings=decode_strings, use_numpy=use_numpy)

    return table_structure
