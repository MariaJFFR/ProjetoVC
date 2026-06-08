from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "data" / "mvtec"


@dataclass
class TestRecord:
    path: Path
    label: int
    defect_type: str


def category_train_good_paths(category):

    train_dir = (
        DATASET_ROOT /
        category /
        "train" /
        "good"
    )

    return sorted(train_dir.glob("*.png"))


def category_test_records(category):

    records = []

    test_root = (
        DATASET_ROOT /
        category /
        "test"
    )

    for defect_dir in sorted(test_root.iterdir()):

        if not defect_dir.is_dir():
            continue

        label = 0 if defect_dir.name == "good" else 1

        for img_path in sorted(
            defect_dir.glob("*.png")
        ):

            records.append(
                TestRecord(
                    path=img_path,
                    label=label,
                    defect_type=defect_dir.name,
                )
            )

    return records


def category_defective_records(category):

    return [
        record
        for record in category_test_records(category)
        if record.label == 1
    ]


def category_defect_types(category):

    return sorted({
        record.defect_type
        for record in category_defective_records(category)
    })