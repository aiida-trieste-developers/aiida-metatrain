"""WorkChain to launch METATrain training for a committee of potentials."""

from aiida import load_profile
from aiida_trainspot.engines.workflows.train import TrainWorkChain

from ..calculations.train import MetaTrainCalculation

load_profile()


class MetaTrainWorkChain(TrainWorkChain):
    """WorkChain to launch METATrain training."""

    _process_class = MetaTrainCalculation

    def restart_from(self, calculation):
        """Resume in place from the previous calculation's remote folder.

        METATrain can restart symlinking the remote scratch directory of the previous calculation,
        avoiding a costly re-upload of datasets/checkpoints already present on the remote computer.

        :param calculation: node from the previous calculation
        """
        self.ctx.inputs.parent_folder = calculation.outputs.remote_folder
