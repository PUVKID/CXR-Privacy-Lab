"""Save fixed, non-cherry-picked image examples locally for a controlled comparison.

Examples are copies for visual inspection, never inputs to training or model selection.
No source image, image label, or published split is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from cxr7_contract import assert_training_scope
from cxr_reid.image_ops import process_image


LEVELS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_names(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return [row["Image Index"] for row in csv.DictReader(handle)]


def _selection(official):
    official = Path(official)
    vr = official / "PatientVerification"
    rr = official / "PatientRetrieval/csv_files"
    nih_candidates = set((vr / "test_list.txt").read_text(encoding="utf-8").split())
    nih_candidates &= set(_csv_names(rr / "test.csv"))
    covid_candidates = set(_csv_names(rr / "covid_data.csv"))
    if not nih_candidates or not covid_candidates:
        raise ValueError("No fixed representative image is present in the official test manifests")
    result = {"NIH": min(nih_candidates), "COVID19": min(covid_candidates)}
    for name in result.values():
        # The original manifests contain filenames, not arbitrary paths.
        if Path(name).name != name or "\\" in name or name in (".", ".."):
            raise ValueError("Unsafe source filename in example selection")
    return result


def _condition(mode, strength):
    if isinstance(strength, bool) or strength not in LEVELS or int(strength) != strength:
        raise ValueError("Example strength must be an integer in the ten declared levels")
    strength = int(strength)
    if mode == "original" and strength == 0:
        return "original_000", "00_original.png"
    if mode not in ("blur", "mosaic"):
        raise ValueError("Examples support original0, blur, or mosaic only")
    ordinal = LEVELS.index(strength) + (1 if mode == "blur" else 11)
    return f"{mode}_{strength:03d}", f"{ordinal:02d}_{mode}_{strength:03d}.png"


def save_condition_example(project, official, output, mode, strength):
    """Return per-dataset source/hash/image paths; repeat calls are idempotent.

    NIH: first lexicographic image shared by the original verification and
    multi-image retrieval test sets. COVID: first lexicographic test CSV image.
    Selection happens before looking at any model result, and is fixed forever
    within a run. A changed source file fails instead of changing the example.
    """
    key, filename = _condition(mode, strength)
    project, output = Path(project), Path(output)
    folder = output / "대표사진"
    index_path = folder / "index.json"
    chosen = _selection(official)
    index = {"selection_rule": "lexicographic manifest filename, fixed before model outcomes",
             "pixel_space": "source image RGB before model resize/normalization",
             "source_images_unchanged": True, "sources": {}, "conditions": {}}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    result = {}
    for dataset, name in chosen.items():
        ds_folder = "NIH_ChestXray14" if dataset == "NIH" else "COVID19"
        source = project / "data" / ds_folder / "images" / name
        current = {"source_filename": name, "source_sha256": _sha(source)}
        prior = index["sources"].get(dataset)
        if prior is not None and prior != current:
            raise ValueError(f"Representative source image changed for {dataset}; 원본 변경")
        index["sources"][dataset] = current
    for dataset, name in chosen.items():
        ds_folder = "NIH_ChestXray14" if dataset == "NIH" else "COVID19"
        source = project / "data" / ds_folder / "images" / name
        target = folder / dataset / filename
        old = index["conditions"].get(key, {}).get(dataset)
        if old is not None and target.is_file():
            if _sha(target) != old["image_sha256"]:
                raise ValueError(f"Saved representative was modified: {target}")
            result[dataset] = old
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
            # Mosaic 0 is an identity condition; image_ops itself requires tile>=1.
            transformed = rgb.copy() if strength == 0 else process_image(rgb, mode, strength)
            temp = target.with_suffix(".png.tmp")
            transformed.save(temp, format="PNG")
            os.replace(temp, target)
            width, height = transformed.size
        result[dataset] = {**index["sources"][dataset],
                           "image_path": target.relative_to(output).as_posix(),
                           "image_sha256": _sha(target), "mode": mode, "strength": int(strength),
                           "width": width, "height": height}
    index["conditions"][key] = result
    temp = index_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, index_path)
    print(f"[EXAMPLE] {mode} {strength}: {folder}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description="Fixed representative photos, one image per dataset and condition")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("original", "blur", "mosaic"), default="original")
    p.add_argument("--strength", type=int, default=0)
    args = p.parse_args()
    seven = Path(__file__).resolve().parent.parent
    assert_training_scope(str(seven), str(args.output), str(args.project))
    if args.official.resolve() != (seven / "official_source").resolve():
        raise ValueError("Only bundled author manifests may select representative images")
    print(json.dumps(save_condition_example(args.project, args.official, args.output,
                                            args.mode, args.strength), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
