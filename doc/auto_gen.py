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


submodpage = """
.. sdg4varselect.{mod}.{submod} module:

{submod}
{xxx}

.. automodule:: sdg4varselect.{mod}.{submod}
   :members:
   :show-inheritance:
"""


classes = {
    "models": (
        ("abstract_model", "AbstractModel"),
        # "abstract_mixed_effects_model",
    ),
    # "algo": ("sto_prox_grad_descent_precond",)
}


modules = {"models": ("abstract_model",)}


def get_tree(module, name):
    if name[:8] == "abstract":
        return f"{module}.abstract"
    return module


here = path.abspath(path.dirname(__file__))
root = f"{here}/source/"

if __name__ == "__main__":
    for mod, values in modules.items():
        for submod in values:

            with open(
                f"{root}generated_sdg4varselect.{mod}.{submod}.rst",
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    submodpage.format(
                        mod=mod,
                        submod=submod,
                        xxx="=" * (len(submod)),
                    )
                )

    # for mod, values in classes.items():
    #     for cla in values:
    #         print(root, mod, cla)

    #         with open(
    #             f"{root}mygenerated_sdg4varselect.{mod}.{cla[0]}.rst",
    #             "w",
    #             encoding="utf-8",
    #         ) as file:
    #             file.write(
    #                 classpage.format(mod=get_tree(mod, cla), cla=cla[1]),
    #                 xcla="=" * (len(cla)),
    #             )
