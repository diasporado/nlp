def correct_input_format(body, field_name):
    return isinstance(body, dict) and \
        field_name in body.keys() and \
        isinstance(body[field_name], list)