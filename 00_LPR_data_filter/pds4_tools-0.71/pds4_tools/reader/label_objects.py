from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from itertools import chain
from xml.etree import ElementTree as ET

from .read_label import read_label

from ..utils.constants import PDS4_NAMESPACES
from ..utils.helpers import xml_to_dict
from ..utils.logging import logger_init

from ..extern import six

# Initialize the logger
logger = logger_init()

#################################


class Label(object):
    """ Stores a PDS4 label or a portion of a PDS4 label.

        This class is similar to an ``ElementTree`` Element; most of the basic attributes and methods
        for Element will also work here. Some additional convenience methods and features are provided
        to make it easier to work with PDS4 labels.

        Parameters
        ----------
        (in general `load` should be used instead of using __init__)

        convenient_root : ``ElementTree`` Element
            Root element of the PDS4 label or label portion, modified to normalize whitespace
            in all element and attribute values containing a single non-whitespace line, to
            strip the default namespace, and to use the default PDS4 prefixes for known namespaces.
        unmodified_root : ``ElementTree`` Element
            Root element of the PDS4 label or label portion.
        convenient_namespace_map : dict, optional
            Keys are the namespace URIs and values are the prefixes for the convenient root of this label.
        unmodified_namespace_map : dict, optional
            Keys are the namespace URIs and values are the prefixes for the unmodified root of this label.
        default_root : str or unicode, optional
            Specifies whether the root element used by default when calling methods and attributes
            in this object will be the convenient_root or unmodified_root. Must be one of
            'convenient' or 'unmodified'. Defaults to convenient.

        Examples
        --------

        To load a label from file,

        >>> lbl = Label()
        >>> lbl.load('/path/to/label.xml')

        To search the entire label for logical_identifier (of which only one is allowed),

        >>> lbl.find('.//logical_identifier)

        To search the entire label for Display Settings in the 'disp' namespace
        (of which there could be multiple),

        >>> display_settings = lbl.findall('.//disp:Display_Settings')

        To search the Display Settings for the top-level element, Display_Direction,

        >>> display_settings.find('disp:Display_Direction')

        See method descriptions for more info.
    """

    def __init__(self, convenient_root=None, unmodified_root=None,
                 convenient_namespace_map=None, unmodified_namespace_map=None, default_root='convenient'):

        # ElementTree root containing read-in XML which was modified before storage
        # (see options set in load() below)
        self._convenient_root = convenient_root

        # ElementTree root containing the unmodified XML
        self._unmodified_root = unmodified_root

        # Dictionaries with keys being the namespace URIs and values being the namespace prefixes
        # for namespaces used this label
        self._convenient_namespace_map = convenient_namespace_map if convenient_namespace_map else {}
        self._unmodified_namespace_map = unmodified_namespace_map if unmodified_namespace_map else {}

        # Set default root (running it through setter)
        self._default_root = None
        self.default_root = default_root

    def __getitem__(self, key):
        """ Obtain subelement.

        Parameters
        ----------
        key : int or slice
            Specifies which subelement to select.

        Returns
        -------
        Label
            Subelement of label.
        """

        return Label(self._convenient_root[key], self._unmodified_root[key],
                     self._convenient_namespace_map, self._unmodified_namespace_map,
                     default_root=self.default_root)

    def __repr__(self):
        """
        Returns
        -------
        str
            A repr string identifying the tag element in a similar way to ``ElementTree`` Element.
        """

        if self._convenient_root is None:
            return super(Label, self).__repr__()
        else:
            return '<Label Element {0} at {1}>'.format(repr(self._convenient_root.tag), hex(id(self)))

    def __len__(self):
        """
        Returns
        -------
        int
            Number of (direct) subelements this label has.
        """
        return len(self.getroot())

    def load(self, filename, default_root='convenient'):
        """ Load Label from PDS4 XML label.

        Parameters
        ----------
        filename : str or unicode
            Filename, including path, to XML label.
        default_root : str or unicode, optional
            Specifies whether the root element used by default when calling methods and attributes in
            this object will be the convenient_root or unmodified_root. Must be one of convenient|unmodified.
            Defaults to convenient.

        Returns
        -------
        None
        """

        self._convenient_root, self._convenient_namespace_map = read_label(filename,
                                            strip_extra_whitespace=True, enforce_default_prefixes=True,
                                            include_namespace_map=True, decode_py2=True)

        self._unmodified_root, self._unmodified_namespace_map = read_label(filename,
                                            strip_extra_whitespace=False, enforce_default_prefixes=False,
                                            include_namespace_map=True, decode_py2=True)

        self.default_root = default_root

    @property
    def text(self):
        """
        Returns
        -------
            Text of element.
        """
        return self.getroot().text

    @property
    def tail(self):
        """
        Returns
        -------
            Tail of element.
        """
        return self.getroot().tail

    @property
    def tag(self):
        """
        Returns
        -------
            Tag of element.
        """
        return self.getroot().tag

    @property
    def attrib(self):
        """
        Returns
        -------
            Attributes of element.
        """
        return self.getroot().attrib

    @property
    def default_root(self):
        """
        Returns
        -------
        str or unicode
            Either 'convenient' or 'unmodified'. Specifies whether the root element used by default when
            calling methods and attributes in this object will be the convenient_root or unmodified_root.
        """

        return self._default_root

    @default_root.setter
    def default_root(self, value):

        if value not in ('convenient', 'unmodified'):
            raise ValueError('Unknown default root for Label: {0}.'.format(value))

        self._default_root = value

    def get(self, key, default=None, unmodified=None):
        """ Gets the element attribute named *key*.

        Uses the same format as ``ElementTree.get``.

        Parameters
        ----------
        key : str or unicode
            Attribute name to select.
        default : optional
            Value to return if no attribute matching *key* is found. Defaults to None.
        unmodified : bool or None
            Looks for key in unmodified ``ElementTree`` root if True, or the convenient one if False.
            Defaults to None, which uses `Label.default_root` to decide.

        Returns
        -------
            Value of the attribute named *key* if it exists, or *default* otherwise.
        """
        return self.getroot(unmodified=unmodified).get(key, default)

    def getroot(self, unmodified=None):
        """ Obtains ``ElementTree`` root Element instance underlying this object.

        Parameters
        ----------
        unmodified : bool or None or None, optional
            If True, returns the unmodified (see `Label` docstring for meaning) ``ElementTree`` Element.
            If False, returns convenient root. Defaults to None, which uses `Label.default_root` to
            decide.

        Returns
        -------
        Element instance
            The ``ElementTree`` root element backing this Label.
        """

        root = self._convenient_root

        if self._resolve_unmodified(unmodified):
            root = self._unmodified_root

        return root

    def find(self, match, namespaces=None, unmodified=None, return_ET=False):
        """ Search for the first matching subelement.

        Uses the same format as ``ElementTree.find``. See `Label` docstring or ``ElementTree.find``
        documentation for examples and supported XPATH description.

        The namespaces found in the label, and those contained in the PDS4_NAMESPACES constant are registered
        automatically. If match contains other namespace prefixes then you must pass in *namespaces*
        parameter specifying the URI for each prefix. In case of duplicate prefixes for the same URI, prefixes
        in the label overwrite those in PDS4_NAMESPACES, and prefixes in *namespaces* overwrite both.

        Parameters
        ----------
        match : str or unicode
            XPATH search string.
        namespaces : dict, optional
            Dictionary with keys corresponding to prefix and values corresponding to URI for namespaces.
        unmodified : bool or None, optional
            Searches unmodified ``ElementTree`` root if True, or the convenient one if False.
            Defaults to None, which uses `Label.default_root` to decide.
        return_ET : bool, optional
            Returns an ``ElementTree`` Element instead of a Label if True. Defaults to False.

        Returns
        -------
        Label, ElementTree Element or None
            Matched subelement, or None if there is no match.
        """

        # Select the proper XML root to look in
        root = self.getroot(unmodified)

        # Append known namespaces if search contains them
        namespaces = self._append_known_namespaces(match, namespaces,  unmodified)

        # Find the matching element
        try:
            found_element = root.find(match, namespaces=namespaces)

        # Implement namespaces support and unicode searching for Python 2.6
        except TypeError:

            if namespaces is not None:
                match = self._add_namespaces_to_match(match, namespaces)

            match = self._unicode_match_to_str(match)

            found_element = root.find(match)

        # If return_ET is not used, find the other matching element
        if not return_ET and found_element is not None:

            unmodified = self._resolve_unmodified(unmodified)

            other_element = self._find_other_element(found_element, unmodified)
            args = [self._convenient_namespace_map, self._unmodified_namespace_map, self._default_root]

            if unmodified:
                label = Label(other_element, found_element, *args)
            else:
                label = Label(found_element, other_element, *args)

            found_element = label

        return found_element

    def findall(self, match, namespaces=None, unmodified=None, return_ET=False):
        """ Search for all matching subelements.

        Uses the same format as ``ElementTree.findall``. See `Label` docstring or ``ElementTree.findall``
        documentation for examples and supported XPATH description.

        The namespaces found in the label, and those contained in the PDS4_NAMESPACES constant are registered
        automatically. If match contains other namespace prefixes then you must pass in *namespaces*
        parameter specifying the URI for each prefix. In case of duplicate prefixes for the same URI, prefixes
        in the label overwrite those in PDS4_NAMESPACES, and prefixes in *namespaces* overwrite both.

        Parameters
        ----------
        match : str or unicode
            XPATH search string
        namespaces : dict, optional
            Dictionary with keys corresponding to prefix and values corresponding to URI for namespaces.
        unmodified : bool or None, optional
            Searches unmodified ``ElementTree`` root if True, or the convenient one if False.
            Defaults to None, which uses `Label.default_root` to decide.
        return_ET : bool, optional
            Returned list contains ``ElementTree`` Elements instead of Labels if True. Defaults to False.

        Returns
        -------
        List[Label or ElementTree Element]
            Matched subelements, or [] if there are no matches.
        """

        # Select the proper XML root to look in
        root = self.getroot(unmodified)

        # Append known namespaces if search contains them
        namespaces = self._append_known_namespaces(match, namespaces, unmodified)

        # Find the matching elements
        try:
            found_elements = root.findall(match, namespaces=namespaces)

        # Implement namespaces support and unicode searching for Python 2.6
        except TypeError:

            if namespaces is not None:
                match = self._add_namespaces_to_match(match, namespaces)

            match = self._unicode_match_to_str(match)

            found_elements = root.findall(match)

        # If return_ET is not used, find the other matching elements
        if not return_ET and found_elements is not None:

            labels = []
            unmodified = self._resolve_unmodified(unmodified)

            for element in found_elements:

                other_element = self._find_other_element(element, unmodified)
                args = [self._convenient_namespace_map, self._unmodified_namespace_map, self._default_root]

                if unmodified:
                    label = Label(other_element, element, *args)
                else:
                    label = Label(element, other_element, *args)

                labels.append(label)

            found_elements = labels

        return found_elements

    def findtext(self, match, default=None, namespaces=None, unmodified=None):
        """ Finds text for the first subelement matching match.

        Uses the same format as ``ElementTree.findtext``. See `Label` docstring or ``ElementTree.findtext``
        documentation for examples and supported XPATH description.

        The namespaces found in the label, and those contained in the PDS4_NAMESPACES constant are registered
        automatically. If match contains other namespace prefixes then you must pass in *namespaces*
        parameter specifying the URI for each prefix. In case of duplicate prefixes for the same URI, prefixes
        in the label overwrite those in PDS4_NAMESPACES, and prefixes in *namespaces* overwrite both.

        Parameters
        ----------
        match : str or unicode
            XPATH search string.
        default : optional
            Value to return if no match is found. Defaults to None.
        namespaces : dict, optional
            Dictionary with keys corresponding to prefixes and values corresponding to URIs for namespace.
        unmodified : bool or None, optional
            Searches unmodified ``ElementTree`` root if True, or the convenient one if False.
            Defaults to None, which uses `Label.default_root` to decide.

        Returns
        -------
            Text of the first matched element, or *default* otherwise.
        """

        found_element = self.find(match, namespaces=namespaces, unmodified=unmodified)

        if found_element is None:
            return default

        else:
            return found_element.text

    def to_string(self, unmodified=True, pretty_print=False):
        """ Generate a string representation of XML label.

        Parameters
        ----------
        unmodified : bool or None, optional
            Generates representation of unmodified ``ElementTree`` root if True, or the convenient one if
            False. None uses `Label.default_root` to decide. Defaults to True.
        pretty_print : bool, optional
            String representation is pretty-printed if True. Defaults to False.

        Returns
        -------
        unicode or str
            String representation of `Label`.
        """

        # Obtain root element, while also effectively deepcopying it such that any modifications made
        # inside this method (e.g. during pretty printing) do not affect the original copy.
        root = ET.fromstring(ET.tostring(self.getroot(unmodified), encoding=str('utf-8')))

        # Adapted from effbot.org/zone/element-lib.htm#prettyprint, modified to properly dedent multi-line
        # values with different indents than used by this method (e.g., each line is indented by 4 spaces
        # but this method uses 2), and to preserve a single extra newline anywhere one or more are present.
        def pretty_format_xml(elem, level=0):

            i = level * '  '

            # Preserve single extra newline at the end of text (e.g. <Field>\n\n<offset>var</offset>...)
            i_text = '\n' + i
            if elem.text:

                if elem.text.count('\n') > 1:
                    i_text = ('\n' * 2) + i

            # Preserve single extra newline at the end of an element (e.g. <Field>var</Field>\n\n<offset>...)
            i_tail = '\n' + i
            if elem.tail:
                if elem.tail.count('\n') > 1:
                    i_tail = ('\n' * 2) + i

            # Remove extra space on each newline in multi-line text values
            if elem.text and elem.text.strip() and (elem.text.count('\n') > 0 or elem.text.count('\r') > 0):
                lines = elem.text.splitlines(True)

                # If each line beginning contains extra space (e.g., it used 4 spaces to align but we use 2)
                # then remove that extra space
                for j, line in enumerate(lines):
                    if line[0:len(i)] == i:
                        lines[j] = line[len(i)-1:]

                elem.text = ''.join(lines)

                # On last line of multi-line string, if it ends in a newline and spaces, remove them
                # and replace with a newline and spaces such that the closing tag is on the same level
                # as the opening tag
                if elem.text.rstrip(' ') != elem.text.rstrip():
                    elem.text = elem.text.rstrip() + '\n' + i

            if len(elem):

                if not elem.text or not elem.text.strip():
                    elem.text = i_text + '  '
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i_tail
                for elem in elem:
                    pretty_format_xml(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i_text
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i_tail

        # Pretty-format root node
        if pretty_print:
            pretty_format_xml(root)

        # Remove newline characters from tail of the root element
        if root.tail is not None:
            root.tail = root.tail.strip('\n\r')

        # Obtain string representation, taking care to register_namespaces()
        self._register_namespaces('register', unmodified)
        string = ET.tostring(root, encoding=str('utf-8')).decode('utf-8')
        self._register_namespaces('unregister', unmodified)

        # Adjust to fix issue that in Python 2.6 it is not possible to specify a null prefix correctly
        if 'xmlns:=' in string:
            string = string.replace('xmlns:=', 'xmlns=')
            string = string.replace('<:', '<')
            string = string.replace('</:', '</')

        # Remove UTF-8 processing instruction from first line on Python 2
        if '<?xml' == string.lstrip()[0:5]:
            string_list = string.splitlines(True)
            string = ''.join(string_list[1:])

        # If label is not pretty printed, then we fix a potential issue with indented labels and ET
        # (The indent for an element is in the .tail attribute of the previous element, but for the
        # root element there is no element above it with a tail. We check what is the indent for the
        # closing tag of the root element and manually insert it for the opening tag.)
        if not pretty_print:
            last_tail = root[-1].tail

            if last_tail is not None:
                num_leading_spaces = len(last_tail) - len(last_tail.rstrip(' '))
                num_leading_tabs = len(last_tail) - len(last_tail.rstrip('\t'))
                string = ' ' * num_leading_spaces + '\t' * num_leading_tabs + string

        return string

    def to_dict(self, unmodified=None, skip_attributes=True, cast_values=False, cast_ignore=()):
        """ Generate an `OrderedDict` representation of XML label.

        Parameters
        ----------
        unmodified : bool or None, optional
            Generates representation of unmodified ``ElementTree`` root if True, or the convenient one if
            False. Defaults to None, which uses `Label.default_root` to decide.
        skip_attributes : bool, optional
            If True, skips adding attributes from XML. Defaults to False.
        cast_values : bool, optional
            If True, float and int compatible values of element text and attribute values will be cast as such
            in the output dictionary. Defaults to False.
        cast_ignore : tuple[str or unicode], optional
            If given, then a tuple of element tags and/or attribute names. If *cast_values* is True, then
            for elements and attributes matching exactly the values in this tuple, values will not be cast.
            Attribute names must be prepended by an '@'. Empty by default.

        Returns
        -------
        OrderedDict
            A dictionary representation of `Label`.
        """

        root = self.getroot(unmodified)
        namespace_map = self.get_namespace_map(unmodified)

        # Achieve equivalent of register_namespaces() via tag_modify
        tag_modify = []
        for uri, prefix in namespace_map.items():
            if prefix.strip():
                tag_modify.append(('{{{0}}}'.format(uri), '{0}:'.format(prefix)))
            else:
                tag_modify.append(('{{{0}}}'.format(uri), ''))

        return xml_to_dict(root, skip_attributes=skip_attributes, cast_values=cast_values,
                           cast_ignore=cast_ignore, tag_modify=tuple(tag_modify))

    def get_namespace_map(self, unmodified=None):
        """

        Parameters
        ----------
        unmodified : bool or None, optional
            If True, returns the namespace map for the unmodified (see `Label` docstring) label. If False,
            it uses the convenient label. Defaults to None, which uses `Label.default_root` to decide.

        Returns
        -------
        dict
            A dict with keys being the namespace URIs and values being the namespace prefixes for this
            label.
        """

        namespace_map = self._convenient_namespace_map

        if self._resolve_unmodified(unmodified):
            namespace_map = self._unmodified_namespace_map

        return namespace_map

    def _resolve_unmodified(self, unmodified):
        """ Resolves *unmodified* to either True, or False.

        Parameters
        ----------
        unmodified : bool or None
            Variable to resolve.

        Returns
        -------
        bool
            If *unmodified* is bool, returns unchanged. If None, uses `Label.default_root` to decide
            whether to return True (if 'unmodified') or False (if 'convenient').
        """

        if isinstance(unmodified, bool):
            return unmodified

        elif unmodified is None:
            return self._default_root == 'unmodified'

        else:
            raise TypeError('Unknown unmodified variable: {0}.'.format(unmodified))

    def _find_other_element(self, element, was_unmodified):
        """
        When using `find` or `findall`, we initially search one of the two ``ElementTree`` representations
        (`_convenient_root` or `_unmodified_root`). The purpose of this method is to find the Element
        in the other representation.

        Parameters
        ----------
        element : Label or ElementTree Element
            Element to find
        was_unmodified : bool
            True if *element* was taken from `_unmodified_root`, False otherwise.


        Returns
        -------
        ``ElementTree`` Element
            Matched element.
        """

        other_element = None

        if was_unmodified:
            root = self.getroot(unmodified=True)
            other_root = self.getroot(unmodified=None)
        else:
            root = self.getroot(unmodified=None)
            other_root = self.getroot(unmodified=True)

        # Loop over all elements in root of element to find its number
        node_number = -1

        for i, child_elem in enumerate(root.getiterator()):

            if element == child_elem:
                node_number = i
                break

        # Loop over all elements of other_root to find the matching node_number
        for i, child_elem in enumerate(other_root.getiterator()):

            if i == node_number:
                other_element = child_elem
                break

        return other_element

    def _register_namespaces(self, action, unmodified):
        """
        Registers or unregisters namespace prefixes via ET's ``register_namespace``. The unregister
        functionality is provided because register_namespace is global, affecting any other ET usage.
        Register and unregister allow anything in-between them to effectively have a local namespace register.

        Parameters
        ----------
        action : str or unicode
            Either 'register' or 'unregister', specifying what action to take. Defaults to 'register'.
        unmodified : bool or None
            If True, uses prefixes taken from the unmodified root, instead of the convenient root,
            to register or unregister namespaces.

        Returns
        -------
        None
        """

        prefixes = ET._namespace_map.values()
        uris = ET._namespace_map.keys()

        namespace_map = self.get_namespace_map(unmodified)

        for uri, prefix in namespace_map.items():

            # Register namespaces
            if action == 'register':

                # Check if namespace prefix or URI already exists, then do not register the namespace if so.
                # This allows local namespaces that have the same prefix as a global prefix but a
                # different URI, which is valid in XML, to stay unique)
                if (prefix in prefixes) or (uri in uris):
                    continue

                # Register the namespace
                ET._namespace_map[uri] = prefix

            # Unregister the namespaces
            else:

                if (prefix in prefixes) and (uri in ET._namespace_map.keys()):
                    del ET._namespace_map[uri]

    def _append_known_namespaces(self, match, namespaces, unmodified):
        """
        Appends known namespaces (i.e., those in the namespace map for this label, and those in the
        constant PDS4_NAMESPACES) to *namespaces* if *match* contains prefixes (signified by the colon).
        In case of duplicate prefixes for the same URI, prefixes in the label overwrite those in
        PDS4_NAMESPACES, and prefixes in *namespaces* overwrite both.

        Parameters
        ----------
        match : str or unicode
            XPATH search string.
        namespaces : dict
            Dictionary with keys corresponding to prefix and values corresponding to URI for namespaces.
        unmodified : bool or None
            If True, uses namespace map created from the unmodified root, instead of the convenient root,
             as part of the known namespaces. If None, uses `Label.default_root` to decide.

        Returns
        -------
        dict
            New namespaces dictionary containing previous *namespaces* as well as PDS4_NAMESPACES,
            and those in the namespace map for this label.
        """

        # Merge PDS4_NAMESPACES and namespaces from this label into a single ``dict``. In case of conflict,
        # the latter take precedence. Additionally, if namespaces in this label have a case where a
        # single prefix refers to multiple URIs (via local prefixes), only one will be kept.
        namespace_map = self.get_namespace_map(unmodified)
        known_namespaces = dict(chain(
                                six.iteritems(PDS4_NAMESPACES),
                                six.iteritems(dict((v, k) for k, v in six.iteritems(namespace_map)))))

        if (':' in match) and (namespaces is None):
            namespaces = known_namespaces

        elif ':' in match:
            namespaces = dict(chain(six.iteritems(known_namespaces), six.iteritems(namespaces)))

        return namespaces

    @classmethod
    def _add_namespaces_to_match(cls, match, namespaces):
        """
        Python 2.6 does not support the namespaces parameter for ``ElementTree``'s `find`, `findall`,
        which are used in the implementation of `Label`'s `find`, `findall` and `findtext`. To add this
        support, in *match* we replace the prefix with the URI for that prefix contained in brackets.
        This is also how ``ElementPath`` works in Python 2.7 and above, from which this code is adapted.

        Parameters
        ----------
        match : str or unicode
            XPATH search string.
        namespaces : dict
            Dictionary with keys corresponding to prefix and values corresponding to URI for namespaces.

        Returns
        -------
        str or unicode
            A new XPATH search string, with prefix replaced by {URI}.

        Examples
        --------
        >>> match = './/disp:Display_Settings'
        >>> namespaces = {'disp': 'http://pds.nasa.gov/pds4/disp/v1', 'sp': 'http://pds.nasa.gov/pds4/sp/v1'}

        >>> match = self._add_namespaces_to_match(match, namespaces)
        >>> print match
        .//{http://pds.nasa.gov/pds4/disp/v1}Display_Settings
        """

        xpath_tokenizer_re = re.compile("("
                                        "'[^']*'|\"[^\"]*\"|"
                                        "::|"
                                        "//?|"
                                        "\.\.|"
                                        "\(\)|"
                                        "[/.*:\[\]\(\)@=])|"
                                        "((?:\{[^}]+\})?[^/\[\]\(\)@=\s]+)|"
                                        "\s+")
        modified_match = ''

        for token in xpath_tokenizer_re.findall(match):
            tag = token[1]

            if tag and tag[0] != '{' and ':' in tag:

                try:
                    prefix, uri = tag.split(":", 1)
                    if not namespaces:
                        raise KeyError

                    modified_match += '{0}{{{1}}}{2}'.format(token[0], namespaces[prefix], uri)
                except KeyError:
                    raise SyntaxError('prefix {0} not found in prefix map'.format(prefix))
            else:
                modified_match += '{0}{1}'.format(*token)

        return modified_match

    @classmethod
    def _unicode_match_to_str(cls, match):
        """
        Python 2.6 has a bug in ``ElementTree``'s `find`, `findall` and `findtext` that affects searching
        for unicode match strings. Specifically, when searching for at least the immediate descendents,
        ``ElementPath`` checks that type("") [which is ``str``] is equivalent to the tag, otherwise it sets
        the tag to none. This breaks certain searches, for example element.findall('.//Unicode_Tag') would
        not find a match if the Unicode_Tag element is a direct descendant of element. However just
        './Unicode_Tag' would work there.

        Parameters
        ----------
        match : str or unicode
            A search string for `find`, `findall` or `findtext`.

        Returns
        -------
        str or unicode
            The same search string as *match*, typecast to ``str`` type if *match* was ASCII-compatible.
        """

        try:
            match = str(match.decode('ascii'))

        except UnicodeError:
            logger.warning('Python 2.6 find, findall and findtext results may exclude valid matches when '
                           'the search string contains unicode characters. Detected unicode search: {0}'
                           .format(match))

        return match


def get_display_settings_for_lid(local_identifier, label):
    """ Search a PDS4 label for Display_Settings of a data structure with local_identifier.

    Parameters
    ----------
    local_identifier : str or unicode
        The local identifier of the data structure to which the display settings belong.
    label : Label or ElementTree Element
        Label for a PDS4 product with-in which to look for the display settings.

    Returns
    -------
    Label, ElementTree Element or None
        Found Display_Settings section with same return type as *label*, or None if not found.
    """

    matching_display = None

    # Find all the Display Settings classes in the label
    displays = label.findall('.//disp:Display_Settings')
    if not displays:
        return None

    # Find the particular Display Settings for this LID
    for display in displays:

        # Look in both PDS and DISP namespace due to standards changes in the display dictionary
        lid_disp = display.findtext('.//disp:local_identifier_reference')
        lid_pds = display.findtext('.//local_identifier_reference')

        if local_identifier in (lid_disp, lid_pds):
            matching_display = display
            break

    return matching_display


def get_spectral_characteristics_for_lid(local_identifier, label):
    """ Search a PDS4 label for Spectral_Characteristics of a data structure with local_identifier.

    Parameters
    ----------
    local_identifier : str or unicode
        The local identifier of the data structure to which the spectral characteristics belong.
    label : Label or ElementTree Element
        Label for a PDS4 product with-in which to look for the spectral characteristics.

    Returns
    -------
    Label,  ElementTree Element or None
        Found Spectral_Characteristics section with same return type as *label*, or None if not found.
    """

    matching_spectral = None

    # Find all the Spectral Characteristics classes in the label
    spectra = label.findall('.//sp:Spectral_Characteristics')
    if not spectra:
        return None

    # Find the particular Spectral Characteristics for this LID
    for spectral in spectra:

        # There may be multiple local internal references for each spectral data object sharing these
        # characteristics. Also look in both PDS and SP namespace due to standards changes in the spectral
        # dictionary
        references = spectral.findall('sp:Local_Internal_Reference') + \
                     spectral.findall('Local_Internal_Reference')

        for reference in references:

            # Look in both PDS and DISP namespace due to standards changes in the display dictionary
            lid_sp = reference.findtext('sp:local_identifier_reference')
            lid_pds = reference.findtext('local_identifier_reference')

            if local_identifier in (lid_sp, lid_pds):
                matching_spectral = spectral
                break

    return matching_spectral


def get_mission_area(label):
    """ Search a PDS4 label for a Mission_Area.

    Parameters
    ----------
    label : Label or ElementTree Element
        Label for a PDS4 product with-in which to look for a mission area.

    Returns
    -------
    Label, ElementTree Element or None
        Found Mission_Area section with same return type as *label*, or None if not found.
    """

    return label.find('.//Mission_Area')


def get_discipline_area(label):
    """ Search a PDS4 label for a Discipline_Area.

    Parameters
    ----------
    label : Label or ElementTree Element
        Label for a PDS4 product with-in which to look for a discipline area.

    Returns
    -------
    Label, ElementTree Element or None
        Found Discipline_Area section with same return type as *label*, or None if not found.
    """
    return label.find('.//Discipline_Area')
