def argument_autofilling(func, kwargs):
    """ Arguments auto-filling for function "func"
    Args:
        func    ('function'): function to be filled.
        kwargs  ('dict'):     input
    Returns:
        option  ('dict'):     output
    """
    option = dict()
    kw = func.__code__.co_varnames
    for i in range(func.__code__.co_argcount):
        argname = kw[i]
        if i == 0 and argname == 'self':
            continue
        if argname in kwargs:
            option[argname] = kwargs[argname]
    return option


def get_attributes(_o: object, _name: str, _default=None):
    attributes = _name.split('.')
    o = _o
    for attribute in attributes:
        o = getattr(o, attribute, None)
        if o is None:
            break
    return o
