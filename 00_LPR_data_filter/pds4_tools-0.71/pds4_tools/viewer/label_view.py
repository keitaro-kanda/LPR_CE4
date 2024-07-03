from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import platform

from .core import Window
from .widgets.tree import TreeView

from ..reader.label_objects import (get_mission_area, get_discipline_area,
                                    get_spectral_characteristics_for_lid, get_display_settings_for_lid)

from ..extern.six.moves.tkinter import Event as TKEvent
from ..extern.six.moves.tkinter import (Menu, Frame, Scrollbar, Label, Entry, Text, Button, Checkbutton,
                                        BooleanVar, StringVar)


class LabelWindow(Window):
    """ Base class; Window used to display the content of a label """

    def __init__(self, viewer, full_label, structure_label=None, initial_display='full label'):

        # Set initial necessary variables and do other required initialization procedures
        super(LabelWindow, self).__init__(viewer)

        # Set instance variables
        self.full_label = full_label
        self.structure_label = structure_label

        # Stores possibilities of what can be displayed (these are listed in the View menu), 'break' is
        # not a valid option but simply indicates a break in the menu should be where it occurs
        self._display_types = ['full label', 'object label', 'break', 'discipline area', 'mission area',
                               'break', 'display settings', 'spectral characteristics']

        # Stores what display type is currently selected
        display_type = StringVar()
        self._menu_options['display_type'] = display_type
        self._add_trace(display_type, 'w', self._update_label, default=initial_display)

        # Stores whether label is being pretty printed or shown as in the file
        pretty_print = BooleanVar()
        self._menu_options['pretty_print'] = pretty_print
        self._add_trace(pretty_print, 'w', self._update_label, default=True)

        # Stores search string in search box
        self._search_text = StringVar()
        self._add_trace(self._search_text, 'w', self._search_label)

        # Stores whether match case box is selected
        self._match_case = BooleanVar()
        self._add_trace(self._match_case, 'w', self._search_label, default=False)

        # Stores a list of 3-valued tuples, each containing the line number, start position
        # and stop position of each result that matches the search string; and stores the index
        # of the tuple for the last shown result
        self._search_match_results = []
        self._search_match_idx = -1

        # Create the menu
        self._add_menu()

        # Create the rest of the LabelWindow content
        self._text_pad = None
        self._search_match_label = None
        self._draw_content()

        # Add notify event for scroll wheel (used to scroll label via scroll wheel without focus)
        self._bind_scroll_event(self._mousewheel_scroll)

        self._center_and_fit_to_content()
        self._show_window()

    # Adds menu options used for manipulating the label display
    def _add_menu(self):

        # Create File Menu
        self._menu = Menu(self._widget)
        self._add_file_menu()

        # Create Edit Menu
        edit_menu = Menu(self._menu, tearoff=0)
        edit_menu.add_command(label='Select All', command=self._select_all)
        edit_menu.add_command(label='Copy', command=self._copy)
        self._menu.add_cascade(label='Edit', menu=edit_menu)

        # Create View menu
        view_menu = Menu(self._menu, tearoff=0)
        self._menu.add_cascade(label='View', menu=view_menu)

        for display_type in self._display_types:

            if display_type == 'break':
                view_menu.add_separator()
                continue

            view_menu.add_checkbutton(label=display_type.title(), onvalue=display_type, offvalue=display_type,
                                      variable=self._menu_options['display_type'])

            if self._label_for_display_type(display_type) is None:
                view_menu.entryconfig(view_menu.index('last'), state='disabled')

    # Draws the majority of the LabelWindow content (header, label text pad, scrollbars and search box)
    def _draw_content(self):

        # Add header
        header_box = Frame(self._widget, bg=self.get_bg('gray'))
        header_box.pack(side='top', fill='x')

        header = Label(header_box, text='Label', bg=self.get_bg('gray'), font=self.get_font(15, 'bold'))
        header.pack(pady=10, side='top')

        # Search box's parent frame (ensures that search box remains in view when window is resized down)
        search_box_parent_frame = Frame(self._widget, bg=self.get_bg('gray'))
        search_box_parent_frame.pack(side='bottom', fill='x', anchor='nw')

        # Add search box
        search_box_frame = Frame(search_box_parent_frame, height=20, bg=self.get_bg('gray'))
        search_box_frame.pack_propagate(False)

        search_box = Entry(search_box_frame, bg='white', bd=0, highlightthickness=0, textvariable=self._search_text)
        search_box.pack(side='left')
        search_box.bind('<Return>', self._search_label)
        search_box.focus()

        search_button = Button(search_box_frame, text='Search', width=7, command=self._search_label,
                               bg=self.get_bg('gray'))
        search_button.pack(side='left', padx=(5, 0))
        search_box_frame.pack(fill='x', padx=(5, 0), pady=5)

        match_case_button = Checkbutton(search_box_frame, text='Match Case', variable=self._match_case,
                                        bg=self.get_bg('gray'))
        match_case_button.pack(side='left', padx=(5, 0))

        self._search_match_label = Label(search_box_frame, fg='slate gray', bg=self.get_bg('gray'))

        # Add text pad
        text_pad_frame = Frame(self._widget)
        self._create_text_pad(text_pad_frame)

        # Add scrollbars for text pad
        vert_scrollbar = Scrollbar(text_pad_frame, orient='vertical', command=self._text_pad.yview)
        self._text_pad.configure(yscrollcommand=vert_scrollbar.set)

        horz_scrollbar = Scrollbar(text_pad_frame, orient='horizontal', command=self._text_pad.xview)
        self._text_pad.configure(xscrollcommand=horz_scrollbar.set)

        vert_scrollbar.pack(side='right', fill='y')
        horz_scrollbar.pack(side='bottom', fill='x')
        self._text_pad.pack(side='left', expand=1, fill='both')
        text_pad_frame.pack(side='top', expand=1, fill='both')

    # Creates text pad
    def _create_text_pad(self, frame):
        return NotImplementedError

    # Sets text shown in text pad for specified label
    def _set_label(self, label):
        return NotImplementedError

    # Searches label for the string in the search box
    def _search_label(self, *args):

        # Get text in search box, and whether match case is selected
        search_text = self._search_text.get().strip('\n\r')
        match_case = self._match_case.get()

        # Remove any previous match, and configure tag for new match
        self._text_pad.tag_delete('search')
        self._text_pad.tag_configure('search', background='yellow')

        # Do not try to search if search string is empty
        if not search_text.strip():
            self._search_match_results = []
            self._update_search_match_label(match_num=0, total_matches=0, action='hide')
            return

        # Start search from beginning unless method was called by an event binding (likely for the enter key)
        # or without arguments (meaning search button was pressed)
        if len(args) != 0 and not isinstance(args[0], TKEvent):

            self._search_match_results = []
            self._search_match_idx = -1

            text_pad_string = self._text_pad.get('1.0', 'end-1c')

            if not match_case:
                text_pad_string = text_pad_string.lower()
                search_text = search_text.lower()

            # Find start and stop positions of each match for search_text. Note that although we could
            # use Text widget's native search, it is extremely slow if the widget has thousands of lines
            # (i.e, for a large label)
            for i, line in enumerate(text_pad_string.splitlines()):

                start_position = 0

                # For each line with at least one match, determine start and stop positions
                # of each match in the line
                for j in range(0, line.count(search_text)):
                    start_position = line.find(search_text, start_position)
                    stop_position = start_position + len(search_text)

                    self._search_match_results.append((i, start_position, stop_position))
                    start_position = stop_position

        # Do not continue if no results found
        num_matches = len(self._search_match_results)

        if num_matches == 0:
            self._update_search_match_label(match_num=0, total_matches=0, action='draw')
            return

        # Find current match
        self._search_match_idx += 1

        if self._search_match_idx >= num_matches:
            self._search_match_idx = 0

        matching_result = self._search_match_results[self._search_match_idx]
        start_index = '{0}.{1}'.format(matching_result[0] + 1, matching_result[1])
        stop_index = '{0}.{1}'.format(matching_result[0] + 1, matching_result[2])

        # Show the match
        self._text_pad.tag_add('search', start_index, stop_index)
        self._text_pad.see(stop_index)

        self._update_search_match_label(match_num=self._search_match_idx, total_matches=num_matches,
                                        action='draw')

    # Updates current display to show whether match was not found for search
    def _update_search_match_label(self, match_num=0, total_matches=0, action='draw'):

        if total_matches > 0:
            matches_text = '({0} of {1} matches)'.format(match_num + 1, total_matches)
        else:
            matches_text = '(No match found)'

        self._search_match_label.config(text=matches_text)

        if action == 'draw':
            self._search_match_label.pack(side='left', padx=(5, 0))

        else:
            self._search_match_label.pack_forget()

    # Updates current display to show selected label (based on display type)
    def _update_label(self, *args):

        display_type = self.menu_option('display_type')

        label = self._label_for_display_type(display_type)
        self._set_label(label)

    # Obtain the correct label based on the label display type, or None if there is no available label
    def _label_for_display_type(self, display_type):

        label = None

        object_lid = None
        if self.structure_label is not None:
            object_lid = self.structure_label.findtext('local_identifier')

        # Retrieve which label should be shown
        if display_type == 'full label':
            label = self.full_label

        elif display_type == 'discipline area':
            label = get_discipline_area(self.full_label)

        elif display_type == 'mission area':
            label = get_mission_area(self.full_label)

        elif self.structure_label is not None:

            if display_type == 'object label':
                label = self.structure_label

            elif display_type == 'display settings':
                label = get_display_settings_for_lid(object_lid, self.full_label)

            elif display_type == 'spectral characteristics':
                label = get_spectral_characteristics_for_lid(object_lid, self.full_label)

        return label

    def _select_all(self):
        self._text_pad.focus()
        self._text_pad.tag_add('sel', '1.0', 'end')

    def _copy(self):

        if self._text_pad.tag_ranges('sel'):
            message = self._text_pad.get('sel.first', 'sel.last')
        else:
            message = ''

        self._widget.clipboard_clear()
        self._widget.clipboard_append(message)

    # Called on mouse wheel scroll action, scrolls label up or down
    def _mousewheel_scroll(self, event):

        event_delta = int(-1 * event.delta)

        if platform.system() != 'Darwin':
            event_delta //= 120

        self._text_pad.yview_scroll(event_delta, 'units')


class LabelTreeViewWindow(LabelWindow):
    """ Window used to display the content of a label as a TreeView """

    def __init__(self, viewer, full_label, structure_label=None, initial_display='full label'):

        # Set initial necessary variables and do other required initialization procedures
        super(LabelTreeViewWindow, self).__init__(viewer, full_label, structure_label,
                                                  initial_display=initial_display)

        # Set a title for the window
        self._set_title("{0} - Label View".format(self._widget.title()))

        # TreeView object, which manages the text pad used to display the label
        self._tree_view = None

        # Display initial label content
        self._update_label()

    # Adds menu options used for manipulating the label display
    def _add_menu(self):

        super(LabelTreeViewWindow, self)._add_menu()

        # Append to View menu
        view_menu = self._menu.winfo_children()[-1]
        view_menu.add_separator()

        view_menu.add_command(label='Show XML Label', command=lambda:
                              open_label(self._viewer, self.full_label, self.structure_label,
                                         initial_display=self.menu_option('display_type'), type='xml'))

    # Create TreeView of the label, which has the text pad
    def _create_text_pad(self, frame):
        self._text_pad = Text(frame, width=100, height=30, wrap='none', relief='flat',
                              highlightthickness=0, bd=0, bg='white')

    # Sets text shown in text pad
    def _set_label(self, label):

        label_dict = label.to_dict()

        # Delete top-level attributes of a full label (e.g. those of Product_Observational)
        if self.menu_option('display_type') == 'full label':

            values = list(label_dict.values())[0]
            for key in values:
                if key[0] == '@':
                    del values[key]
                    break

        # Clear text pad of any current content
        self._text_pad.config(state='normal')
        self._text_pad.delete('1.0', 'end')
        self._text_pad.see('0.0')
        self._text_pad.tag_delete(self._text_pad.tag_names())
        self._text_pad.config(state='disabled')

        # Remove focus from text pad if it has it. For large labels (those having a large amount of text),
        # inserting new text into an existing text pad seems to be extremely slow if it has focus
        if self._text_pad.focus_get() == self._text_pad:
            self._widget.focus()
            self._widget.update()

        # Create the TreeView for label
        self._tree_view = TreeView(self._text_pad, label_dict,
                                   header_font=self.get_font(weight='bold'),
                                   key_font=self.get_font(weight='underline'),
                                   value_font=self.get_font(name='monospace'),
                                   spacing_font=self.get_font())

    # Searches label for the string in the search box
    def _search_label(self, *args):

        # Search the label
        super(LabelTreeViewWindow, self)._search_label(*args)

        # Do not continue if no results found
        if len(self._search_match_results) == 0:
            return

        matching_result = self._text_pad.tag_ranges('search')

        # Maximize element and any necessary parents of element such that result can be seen
        for tag_name in self._text_pad.tag_names(matching_result[0]):
            element = self._tree_view.find_element_by_id(tag_name)

            if element is not None:

                for parent in reversed(element.parents()):
                    parent.maximize()

                break

        # Show the match
        self._text_pad.see(matching_result[1])


class LabelXMLWindow(LabelWindow):
    """ Window used to display the content of a label as XML """

    def __init__(self, viewer, full_label, structure_label=None, initial_display='full label'):

        # Set initial necessary variables and do other required initialization procedures
        super(LabelXMLWindow, self).__init__(viewer, full_label, structure_label,
                                             initial_display=initial_display)

        # Set a title for the window
        self._set_title("{0} - Label XML View".format(self._widget.title()))

        # Display initial label content
        self._update_label()

    # Adds menu options used for manipulating the label display
    def _add_menu(self):

        super(LabelXMLWindow, self)._add_menu()

        # Create Options Menu
        options_menu = Menu(self._menu, tearoff=0)
        self._menu.insert_cascade(self._menu.index('last'), label='Options', menu=options_menu)

        options_menu.add_checkbutton(label='Format Label', onvalue=True, offvalue=False,
                                     variable=self._menu_options['pretty_print'])
        options_menu.add_checkbutton(label="Initial Label", onvalue=False, offvalue=True,
                                     variable=self._menu_options['pretty_print'])

    def _create_text_pad(self, frame):
        self._text_pad = Text(frame, width=100, height=30, wrap='none', background='white',
                              borderwidth=0, highlightthickness=0)

    # Sets text shown in text pad
    def _set_label(self, label):

        # Retrieve whether label should be formatted or not
        if self.menu_option('pretty_print'):
            label_text = label.to_string(pretty_print=True)
        else:
            label_text = label.to_string(pretty_print=False)

        self._text_pad.config(state='normal')
        self._text_pad.delete('1.0', 'end')
        self._text_pad.see('0.0')
        self._text_pad.insert('1.0', label_text)
        self._text_pad.config(state='disabled')


#  Opens a new LabelWindow, either for TreeView or for XMLView
def open_label(viewer, full_label, structure_label=None, initial_display=None, type='tree'):

    args = [viewer, full_label]
    kwargs = {'structure_label': structure_label, 'initial_display': initial_display}

    if type == 'tree':
        label_view = LabelTreeViewWindow(*args, **kwargs)

    else:
        label_view = LabelXMLWindow(*args, **kwargs)

    return label_view
