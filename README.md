# aiida-metatrain

AiiDA plugin to run [metatrain](https://github.com/metatensor/metatrain) trainings. It provides a
calculation, a parser and a work chain built on top of the generic training engine of
[aiida-trains-pot](https://github.com/aiida-trieste-developers/aiida-trains-pot)
(`aiida_trainspot`).

## Requirements

- `aiida-core >= 2.6.2, < 3`
- `aiida-trainspot >= 2.0.0`
- metatrain installed on the target computer, with the `mtt` executable configured as an AiiDA
  code.

## Installation

```bash
pip install .
```

## Entry points

| Entry point      | Class                  | Type        |
| ---------------- | ---------------------- | ----------- |
| `metatrain`      | `MetaTrainCalculation` | Calculation |
| `metatrain`      | `MetaBaseParser`       | Parser      |
| `metatrain.base` | `MetaTrainWorkChain`   | Work chain  |

## Calculation

`MetaTrainCalculation` writes the metatrain configuration file and runs the training.

Main inputs, in addition to those of the base `TrainCalculation`
(`training_set`, `validation_set`, `test_set`, `checkpoints`, `postprocess_code`, `restart`):

- `meta_config` (`Dict`, optional): metatrain configuration. Unless `finetune_model` is given,
  the `training_set`, `validation_set` and `test_set` sections are filled in by the plugin to
  point at the extended XYZ files it writes, reading the `dft_energy` key as the energy target.
- `finetune_model` (`SinglefileData`, optional): checkpoint to finetune.
- `parent_folder` (`RemoteData`, optional): remote folder of a previous training. Datasets and
  checkpoints are symlinked instead of being uploaded again. It must be on the same computer as
  the code.

Datasets are written as extended XYZ with the `dft_energy` key. If a `checkpoints` folder is
given, training restarts from it.

Output: `model` (`SinglefileData`, the trained model), plus `checkpoints` and `RMSE` from the
base class.

## Work chain

`MetaTrainWorkChain` extends `TrainWorkChain` and handles restarts. When a calculation runs out
of walltime it is restarted from the remote folder of the previous run, so datasets and
checkpoints already present on the remote computer are reused.

## Acknowledgments

This project was supported by:

- **[Università degli Studi di Trieste](https://portale.units.it/en)**
- **[Scuola Internazionale Superiore di Studi Avanzati](https://www.sissa.it/it)**
- **[Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing](https://www.supercomputing-icsc.it/en/icsc-home/)**
- **[EU Centre of Excellence "MaX – Materials Design at the Exascale"](https://www.max-centre.eu/)**
- **POC Programma Operativo Complementare al POR FSE Friuli Venezia Giulia 2014-2020**
