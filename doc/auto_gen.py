# pylint: skip-file
from os import path

classpage = """
.. {mod}.{cla}:

{cla}
{xxx}

.. currentmodule:: sdg4varselect.{mod}

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
        ("abstract_model", "AbstractLatentVariablesModel"),
        ("abstract_model", "AbstractMixedEffectsModel"),
        ("abstract_model", "AbstractHDModel"),
    ),
    # "algo": ("sto_prox_grad_descent_precond",)
}


modules = {"models": ("abstract_model",)}


def get_tree(module, name):
    if name[:8] == "abstract":
        return f"{module}.abstract"
    return module


here = path.abspath(path.dirname(__file__))
root = f"{here}/source/mygenerated/"

if __name__ == "__main__":
    # for mod, values in modules.items():
    #     for submod in values:

    #         with open(
    #             f"{root}sdg4varselect.{mod}.{submod}.rst",
    #             "w",
    #             encoding="utf-8",
    #         ) as file:
    #             file.write(
    #                 submodpage.format(
    #                     mod=mod,
    #                     submod=submod,
    #                     xxx="=" * (len(submod)),
    #                 )
    #             )

    for mod, values in classes.items():
        for cla in values:

            with open(
                f"{root}/{mod}/sdg4varselect.{mod}.{cla[1]}.rst",
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    classpage.format(
                        mod=mod,
                        cla=cla[1],
                        xxx="=" * len(cla[-1]),
                    )
                )
