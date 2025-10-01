Installation
============

How to install the package
--------------------------

You can install the package in two different ways:

1. From the latest release available on GitHub.
2. From the source code (development mode).

1. Install from the latest release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``pip`` to install directly from the GitHub repository::

   pip install git+https://github.com/caillebotteAntoine/sdg4varselect.git@v0.1.0

This will install the package along with its dependencies.

2. Install from source (development mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute or explore the source code, you can install the package manually.

.. Note::
    The sdg4varselect repository use `Poetry` to manage dependencies. You will need to install it first if you want to install the package from source.
    For more information on how to install and use `Poetry`, please refer to the official documentation: https://python-poetry.org/docs/.



Install Poetry, if this has not already been done
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   curl -sSL https://install.python-poetry.org | python3 -

Clone the repository
^^^^^^^^^^^^^^^^^^^^
::

   git clone https://github.com/caillebotteAntoine/sdg4varselect

Go inside the project
^^^^^^^^^^^^^^^^^^^^^
::

   cd sdg4varselect

Install dependencies with Poetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

   poetry install --no-root
