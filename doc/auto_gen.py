# pylint: skip-file
from os import path

classpage = """
.. sdg4varselect.{mod}.{cla}:

{cla}


.. autoclass:: sdg4varselect.{mod}.{cla}
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
"""

# {xcla}
classes = {"models": ("abstract_model",), "algo": ("sto_prox_grad_descent_precond",)}


def get_tree(module, name):
    if name[:8] == "abstract":
        return f"{module}.abstract"
    return module


here = path.abspath(path.dirname(__file__))
root = f"{here}/source/generated"

if __name__ == "__main__":

    for mod, values in classes.items():
        for cla in values:
            print(root, mod, cla)

            with open(
                f"{root}/sdg4varselect.{mod}.{cla}.rst", "w", encoding="utf-8"
            ) as file:
                file.write(
                    classpage.format(mod=get_tree(mod, cla), cla=cla)
                )  # , xcla="=" * (len(cla))))
