"""AdEx sweep entry point. See HPC_SIMULATIONS.md for usage and storage semantics."""

from snn_full_common import sweep_main


if __name__ == "__main__":
    sweep_main("AdEx", __file__)
