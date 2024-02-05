from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())

TASKS = [
    "glue", "super_glue"
]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS