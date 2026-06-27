"""
H2 — Learned representations transfer better across organisms.

Protocol:
1. Pretrain JEPA on zebrafish calcium imaging data.
2. Fine-tune on Drosophila data with a limited budget.
3. Compare against a supervised baseline transferred with the same budget.

Required config keys:
    zebrafish_data_dir : str  — path to zebrafish TIFF stacks (unlabeled pretrain)
    drosophila_data_dir: str  — path to Drosophila labeled data (LabeledTIFFDataset layout)
    finetune_budget    : int  — number of fine-tuning epochs for transfer (default 10)

MLflow tags: hypothesis=H2, source_organism=zebrafish, target_organism=drosophila,
             mode={pretrained|supervised_baseline}
"""

from neuroseg.models.state import State


def run_h2(state: State):
    raise NotImplementedError(
        "H2 trainer is not yet implemented. "
        "Provide zebrafish_data_dir and drosophila_data_dir in config."
    )
