"""AiiDA calculation plugin for the METATrain code."""

import copy
import os

import yaml

from aiida.common import datastructures
from aiida.orm import Dict, RemoteData, SinglefileData
from aiida_trainspot.engines.calculations.train import TrainCalculation


class MetaTrainCalculation(TrainCalculation):
    """AiiDA calculation plugin for the METATrain code."""

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        spec.inputs["metadata"]["options"]["parser_name"].default = "metatrain"

        spec.input("meta_config", valid_type=Dict, help="Config parameters for METATrain", required=False)
        spec.input(
            "finetune_model",
            valid_type=SinglefileData,
            help="Model to finetune",
            required=False,
        )
        spec.input(
            "parent_folder",
            valid_type=RemoteData,
            required=False,
            help="Remote folder of a previous METATrain training to reuse datasets/checkpoints",
        )

        spec.output(
            "model",
            valid_type=SinglefileData,
            help="Model",
        )

    @staticmethod
    def validate_inputs(inputs, ctx):
        """Validate the top-level inputs, extending the common checks with the METATrain-specific ones."""
        error = TrainCalculation.validate_inputs(inputs, ctx)
        if error:
            return error

        parent_folder = inputs.get("parent_folder")

        if parent_folder and parent_folder.computer.uuid != inputs["code"].computer.uuid:
            return "parent_folder must be on the same computer as the code."

    def prepare_for_submission(self, folder):
        """Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        meta_config_dict = self.inputs.meta_config.get_dict()
        parent_folder = self.inputs.get("parent_folder", None)

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = """train config.yml""".split()

        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = "meta.out"

        if parent_folder is None:
            training_txt = self.inputs.training_set.get_txt(write_params=False, key_prefix="dft")
            validation_txt = self.inputs.validation_set.get_txt(write_params=False, key_prefix="dft")
            test_txt = self.inputs.test_set.get_txt(write_params=False, key_prefix="dft")

            with folder.open("training.xyz", "w") as handle:
                handle.write(training_txt)
            with folder.open("validation.xyz", "w") as handle:
                handle.write(validation_txt)
            with folder.open("test.xyz", "w") as handle:
                handle.write(test_txt)

        if "finetune_model" not in self.inputs:
            if "training_set" not in meta_config_dict:
                meta_config_dict["training_set"] = {}

            if "systems" not in meta_config_dict["training_set"]:
                meta_config_dict["training_set"]["systems"] = {}

            if "targets" not in meta_config_dict["training_set"]:
                meta_config_dict["training_set"]["targets"] = {}

            if "energy" not in meta_config_dict["training_set"]["targets"]:
                meta_config_dict["training_set"]["targets"]["energy"] = {}

            meta_config_dict["training_set"]["systems"]["read_from"] = "training.xyz"
            meta_config_dict["training_set"]["systems"]["reader"] = "ase"
            meta_config_dict["training_set"]["systems"]["length_unit"] = "angstrom"

            meta_config_dict["training_set"]["targets"]["energy"]["read_from"] = "training.xyz"
            meta_config_dict["training_set"]["targets"]["energy"]["reader"] = "ase"
            meta_config_dict["training_set"]["targets"]["energy"]["key"] = "dft_energy"
            meta_config_dict["training_set"]["targets"]["energy"]["unit"] = "eV"

            if "test_set" not in meta_config_dict:
                meta_config_dict["test_set"] = copy.deepcopy(meta_config_dict["training_set"])

            meta_config_dict["test_set"]["systems"]["read_from"] = "test.xyz"
            meta_config_dict["test_set"]["targets"]["energy"]["read_from"] = "test.xyz"

            if "validation_set" not in meta_config_dict:
                meta_config_dict["validation_set"] = copy.deepcopy(meta_config_dict["training_set"])

            meta_config_dict["validation_set"]["systems"]["read_from"] = "validation.xyz"
            meta_config_dict["validation_set"]["targets"]["energy"]["read_from"] = "validation.xyz"

        finetune = False
        if "finetune_model" in self.inputs:
            finetune = True

        with folder.open("config.yml", "w") as yaml_file:
            yaml.dump(meta_config_dict, yaml_file, default_flow_style=False)

        calcinfo = datastructures.CalcInfo()
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = []

        # Save the checkpoints folder
        if "checkpoints" in self.inputs:
            checkpoints_folder = self.inputs.checkpoints
            folder.get_subfolder("checkpoints", create=True)  # Create the checkpoints directory
            for checkpoint_file in checkpoints_folder.list_object_names():
                with checkpoints_folder.open(checkpoint_file, "rb") as source:
                    with folder.open("checkpoints/model.ckpt", "wb") as destination:
                        destination.write(source.read())

        if parent_folder is not None:
            remote_path = parent_folder.get_remote_path()
            computer_uuid = parent_folder.computer.uuid
            calcinfo.remote_symlink_list.extend(
                [
                    (computer_uuid, os.path.join(remote_path, "training.xyz"), "training.xyz"),
                    (computer_uuid, os.path.join(remote_path, "validation.xyz"), "validation.xyz"),
                    (computer_uuid, os.path.join(remote_path, "test.xyz"), "test.xyz"),
                    (computer_uuid, os.path.join(remote_path, "checkpoints"), "checkpoints"),
                ]
            )

        if "checkpoints" in self.inputs or parent_folder is not None:
            codeinfo.cmdline_params = """train config.yml --restart checkpoints/model.ckpt""".split()

        if finetune:
            calcinfo.local_copy_list = [
                (
                    self.inputs.finetune_model.uuid,
                    self.inputs.finetune_model.filename,
                    "finetune_model.ckpt",
                ),
            ]

        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = ["model.ckpt", "meta.out", "model.pt", "_scheduler-std*", "outputs"]

        return calcinfo
