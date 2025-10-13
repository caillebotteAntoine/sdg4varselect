"""
Utility functions for inheriting docstrings in class hierarchies.
"""

import inspect


def inherit_docstring_from(from_class):
    """
    Decorator that copies the docstring of a method with the same name
    from a parent class if the decorated method has no docstring.

    Parameters
    ----------
    from_class : type
        The class from which to inherit the docstring.

    Returns
    -------
    callable
        A decorator that updates the decorated method's docstring.

    Notes
    -----
    - Only replaces the docstring if the target method has none.
    """

    def decorator(
        method,
    ):  # pylint: disable=missing-return-doc, missing-return-type-doc
        if method.__doc__:
            return method  # Keep existing docstring

        # Look for a docstring in the parent class
        parent_method = getattr(from_class, method.__name__, None)
        if parent_method:
            # Get the unbound function if it's a descriptor (method/property)
            parent_doc = inspect.getdoc(parent_method)
            if parent_doc:
                method.__doc__ = parent_doc
        return method

    return decorator


def inherit_docstring(cls):
    """
    Inherit docstrings from parent class methods.

    Parameters
    ----------
    cls : type
        The class whose methods will inherit docstrings from parent classes.

    Returns
    -------
    type
        The class with updated method docstrings.
    """
    decs = [inherit_docstring_from(base) for base in cls.__bases__]
    for name, method in cls.__dict__.items():
        if callable(method) and not method.__doc__:
            for dec in decs:
                decorated = dec(method)
                if decorated.__doc__:
                    setattr(cls, name, decorated)
                    break
    return cls


if __name__ == "__main__":

    class Parent:
        """A parent class."""

        def greet(self):
            """Say hello."""
            print("Hello")

        def farewell(self):
            """Say goodbye."""
            print("Goodbye")

    class Parent2:
        """Another parent class."""

        def greet(self):
            """Say hi."""
            print("Hi")

        def farewell(self):
            """Say bye."""
            print("Bye")

        def speech(self):
            """Make a speech."""
            print("Speech")

    @inherit_docstring
    class Child(Parent, Parent2):
        """A child class inheriting docstrings."""

        def greet(self):
            print("Hi")

        def farewell(self):
            """Say bye bye."""
            print("bye bye")

        def speech(self):
            print("Speech !")

    print(Child.greet.__doc__)
    print(Child.farewell.__doc__)
    print(Child.speech.__doc__)
