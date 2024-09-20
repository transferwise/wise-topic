import ipywidgets as widgets


def create_collapsible_item(item, level=0):
    """
    Create a collapsible widget to display a list or a dictionary.
    """
    if isinstance(item, dict):
        return display_drilldown(item, level)
    elif isinstance(item, list):
        return create_collapsible_list(item, level)
    else:
        return widgets.Label(f"{item}")


def display_drilldown(d, level=0):
    """
    Recursively create a collapsible widget to display a nested dictionary.
    """
    items = []
    for key, value in d.items():
        if isinstance(value, (dict, list)):
            # Create a collapsible widget for nested dictionaries or lists
            sub_items = create_collapsible_item(value, level + 1)
            accordion = widgets.Accordion(children=[sub_items])
            accordion.set_title(0, key)
            items.append(accordion)
        else:
            # Display key-value pairs as labels
            items.append(widgets.Label(f"{key}: {value}"))

    return widgets.VBox(items)


def create_collapsible_list(lst, level=0):
    """
    Create a collapsible widget to display a list without "Item X" labels.
    """
    items = []
    for value in lst:
        if isinstance(value, (dict, list)):
            # Create a collapsible widget for nested dictionaries or lists within the list
            sub_items = create_collapsible_item(value, level + 1)
            accordion = widgets.Accordion(children=[sub_items])
            accordion.set_title(0, f"List item")
            items.append(accordion)
        else:
            # Display list items directly without labels
            items.append(widgets.Label(f"{value}"))

    return widgets.VBox(items)


# This is how you'd use this in a notebook:
# # Example usage with a nested dictionary
# nested_dict = {
#     "level1": {
#         "level2a": {
#             "level3a": {"value1": 1, "value2": ["blah1", "blah2"]},
#             "level3b": {"value3": 3},
#         },
#         "level2b": {"level3c": {"value4": ["Foo", "bar"]}},
#     }
# }
#
#
#
# from IPython.display import display
# # Create and display the collapsible dictionary widget
# collapsible_widget = display_drilldown(nested_dict)
# display(collapsible_widget)
