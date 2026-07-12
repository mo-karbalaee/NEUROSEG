# NEUROSEG Documentation

Technical and scientific documentation for the NEUROSEG neural segmentation pipeline.

## Documents

| Document | What it covers |
| -------- | -------------- |
| [Overview](overview.md) | Scientific background, problem statement, the three experimental hypotheses, dataset description, related work |
| [Architecture](architecture.md) | JEPA model: ResNet5 encoder, ResUNet predictor, VCLoss regulariser, segmentation head, checkpoint formats |
| [Pipeline](pipeline.md) | LangGraph StateGraph: node descriptions, routing logic, state schema, how training and inference are unified |
| [Experiments](experiments.md) | H1, H2, H3 protocols in full: motivation, data splits, training steps, expected results, MLflow tags |
| [Training](training.md) | Training loops, H1Config, MLflow logging, reproducibility, how to add a new hypothesis |
| [Data](data.md) | Neurofinder format, dataset classes (TIFFVideoDataset, LabeledTIFFDataset, NeurofinderDataset), clip segmentation, demo data |
| [Metrics](metrics.md) | Dice score, mIoU, H3 cosine similarity gap — definitions, edge cases, where each is used |
| [Configuration](configuration.md) | All CLI flags, all YAML config keys, demo config, HPC and notebook settings |
| [HPC](hpc.md) | Running training on FAU's HPC (NHR@FAU / Alex): access, Slurm job scripts, environment setup, data staging |
| [Codebase](codebase.md) | Directory layout, module responsibilities, dependency graph, entry points, test coverage |

## Quick orientation

If you are new to the project, read in this order:

1. **[Overview](overview.md)** — understand the scientific problem and what the three hypotheses are testing.
2. **[Pipeline](pipeline.md)** — understand how data flows through the system.
3. **[Architecture](architecture.md)** — understand the JEPA model.
4. **[Experiments](experiments.md)** — understand what each `--H1 / --H2 / --H3` flag actually does.
5. **[Codebase](codebase.md)** — navigate the source code.

## Reproduce in 20 minutes

See the [README](../README.md#reproduce-in-20-minutes) for the one-command demo:

```bash
git clone https://github.com/MohammadKarbalaee/NEUROSEG.git
cd NEUROSEG
uv sync
bash demo.sh
```
