"""Generate an honest paired main table from historical counts or new rollouts."""

import csv
import json
from pathlib import Path
from .configuration import ASSETS, load_selection
from .metrics import compare, task_row


def reference_rows():
    groups = {"base": [], "stable_interfaces": []}
    with (ASSETS / "reference_tasks.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = "base" if row["variant"] == "native" else row["variant"]
            groups[name].append(
                dict(
                    task=row["task"],
                    success=row["success"] == "True",
                    **{
                        key: int(row[key])
                        for key in (
                            "steps",
                            "generated_tokens",
                            "actual_decode_kv",
                            "actual_prefill_pairs",
                        )
                    },
                )
            )
    return groups


def read_runs(specifications):
    from .runtime.protocol import validate_score

    groups, signatures = {}, {}
    for spec in specifications:
        if "=" not in spec:
            raise ValueError("use --run NAME=DIRECTORY")
        name, directory = spec.split("=", 1)
        if not name:
            raise ValueError("method name is empty")
        root = Path(directory)
        status = json.loads((root / "status.json").read_text())
        meta = json.loads((root / "run.json").read_text())
        if status.get("phase") != "complete":
            raise ValueError(f"incomplete run: {directory}")
        signature = {
            key: meta[key]
            for key in (
                "variant",
                "plan",
                "runtime",
                "generation",
                "bundle_tensor_sha256",
                "model_config_sha256",
                "source_sha256",
            )
        }
        if name in signatures and signature != signatures[name]:
            raise ValueError(f"cannot mix configurations/code versions for {name}")
        signatures[name] = signature
        rows = []
        for path in sorted((root / "trajectories").glob("*.json")):
            record = json.loads(path.read_text())
            validate_score(record["official_score"], "test_normal")
            if record["split"] != "test_normal" or record.get("native") != (
                meta["variant"] == "base"
            ):
                raise ValueError("trajectory method/split mismatch")
            if (
                record.get("merge_plan") != meta["plan"]
                or record["behavior"].get("runtime") != meta["runtime"]
            ):
                raise ValueError("trajectory policy/runtime mismatch")
            rows.append(task_row(record))
        if len(rows) != len(meta["tasks"]) or {r["task"] for r in rows} != set(meta["tasks"]):
            raise ValueError("run is incomplete or contains unexpected trajectories")
        groups.setdefault(name, []).extend(rows)
    if "base" in signatures and signatures["base"]["variant"] != "base":
        raise ValueError("the base label must refer to an uncompressed baseline")
    if signatures:
        baseline = signatures.get("base", next(iter(signatures.values())))
        for signature in signatures.values():
            if any(
                signature[key] != baseline[key]
                for key in ("runtime", "generation", "model_config_sha256")
            ):
                raise ValueError("methods use different base models or evaluation settings")
    return groups


def markdown(totals):
    lines = [
        "| Method | Success ↑ | Steps ↓ | Decode KV / Base ↓ | KV / Success (10⁸) ↓ | Attention Pairs / Base ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(totals, key=lambda key: (key != "base", key)):
        row = totals[name]
        cost = (
            f"{row['kv_per_success'] / 1e8:.3f}"
            if row["kv_per_success"] is not None
            else "undefined (0 successes)"
        )
        lines.append(
            f"| {name} | {row['success']}/{row['tasks']} ({row['success_rate']:.2%}) | "
            f"{row['mean_steps']:.2f} | {row['decode_kv_vs_base']:.2%} | {cost} | {row['attention_pairs_vs_base']:.2%} |"
        )
    return "\n".join(lines)


def report(args):
    if args.reference == bool(args.run):
        raise ValueError("choose --reference OR one or more --run arguments")
    groups = reference_rows() if args.reference else read_runs(args.run)
    if not args.allow_partial:
        expected = set(load_selection()["task_ids"])
        if any({row["task"] for row in rows} != expected for rows in groups.values()):
            raise ValueError(
                "all 168 tasks required; use --allow-partial explicitly for diagnostics"
            )
    totals = compare(groups)
    note = (
        "Paired 168-task test-normal reference: native Base and the frozen Stage-3 method."
        if args.reference
        else "Fresh runtime results; stages=[3]."
    )
    value = dict(
        results=totals,
        note=note,
        cost_scope="Full Attention logical position pairs; not total model FLOPs or measured HBM traffic.",
        amortization="All successful and failed task costs divided by the number of successes.",
        ratios="Cost / Base: 100% is baseline; smaller is better. Reduction = 1 - ratio.",
    )
    print(note + "\n\n" + markdown(totals))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
