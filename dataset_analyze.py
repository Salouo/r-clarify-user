from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_PATH = Path("data/processed_data_expanded.json")
DEFAULT_OUTPUT_PATH = Path("outputs/dataset_analysis/label_count_distribution.pdf")
DEFAULT_LABEL_FIELD = "reflective_action"
DEFAULT_IGNORED_FIELDS = {
    "index",
    "reflective_action",
    "reflective_actions",
    "label_action_id",
}


def load_samples(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{dataset_path} must contain a list field named 'samples'.")

    return samples


def iter_label_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [stable_json(label) for label in value]
    return [stable_json(value)]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sample_group_key(
    sample: dict[str, Any],
    *,
    label_field: str,
    ignored_fields: set[str],
    group_by: str | None,
) -> str:
    if group_by:
        if group_by not in sample:
            raise KeyError(f"Group field '{group_by}' is missing in sample index {sample.get('index')}.")
        return stable_json(sample[group_by])

    fields_to_ignore = set(ignored_fields)
    fields_to_ignore.add(label_field)
    sample_without_label = {
        key: value for key, value in sample.items() if key not in fields_to_ignore
    }
    return stable_json(sample_without_label)


def count_labels_per_sample(
    samples: list[dict[str, Any]],
    *,
    label_field: str,
    ignored_fields: set[str],
    group_by: str | None,
) -> tuple[Counter[int], int, int]:
    labels_by_sample: defaultdict[str, set[str]] = defaultdict(set)
    rows_without_label = 0

    for sample in samples:
        group_key = sample_group_key(
            sample,
            label_field=label_field,
            ignored_fields=ignored_fields,
            group_by=group_by,
        )
        labels = list(iter_label_values(sample.get(label_field)))

        if not labels:
            rows_without_label += 1
            labels_by_sample[group_key]
            continue

        labels_by_sample[group_key].update(labels)

    distribution = Counter(len(labels) for labels in labels_by_sample.values())
    return distribution, len(labels_by_sample), rows_without_label


def nice_tick_step(max_value: int, target_ticks: int = 5) -> int:
    if max_value <= 0:
        return 1

    raw_step = max_value / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if raw_step <= step:
            return int(step)

    return int(10 * magnitude)


def pdf_number(value: float) -> str:
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def set_stroke(hex_color: str) -> str:
    r, g, b = rgb(hex_color)
    return f"{r:.3f} {g:.3f} {b:.3f} RG"


def set_fill(hex_color: str) -> str:
    r, g, b = rgb(hex_color)
    return f"{r:.3f} {g:.3f} {b:.3f} rg"


def estimate_text_width(text: str, font_size: float, *, bold: bool = False) -> float:
    factor = 0.56 if bold else 0.52
    return len(text) * font_size * factor


def add_text(
    commands: list[str],
    text: str,
    x: float,
    y: float,
    *,
    font_size: float,
    color: str = "#1f2933",
    bold: bool = False,
    align: str = "left",
    rotate_90: bool = False,
) -> None:
    font = "/F1B" if bold else "/F1"
    text_width = estimate_text_width(text, font_size, bold=bold)

    if rotate_90:
        if align == "center":
            y -= text_width / 2
        elif align == "right":
            y -= text_width
        matrix = f"0 1 -1 0 {pdf_number(x)} {pdf_number(y)} Tm"
    else:
        if align == "center":
            x -= text_width / 2
        elif align == "right":
            x -= text_width
        matrix = f"1 0 0 1 {pdf_number(x)} {pdf_number(y)} Tm"

    commands.extend(
        [
            "BT",
            set_fill(color),
            f"{font} {pdf_number(font_size)} Tf",
            matrix,
            f"({pdf_escape(text)}) Tj",
            "ET",
        ]
    )


def add_line(
    commands: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
) -> None:
    commands.extend(
        [
            set_stroke(color),
            f"{pdf_number(width)} w",
            f"{pdf_number(x1)} {pdf_number(y1)} m",
            f"{pdf_number(x2)} {pdf_number(y2)} l",
            "S",
        ]
    )


def add_rect(
    commands: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
) -> None:
    commands.extend(
        [
            set_fill(color),
            f"{pdf_number(x)} {pdf_number(y)} {pdf_number(width)} {pdf_number(height)} re",
            "f",
        ]
    )


def write_pdf_page(output_path: Path, commands: list[str], width: int, height: int) -> None:
    stream = ("\n".join(commands) + "\n").encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R /F1B 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(f.tell())
            f.write(f"{i} 0 obj\n".encode("ascii"))
            f.write(obj)
            f.write(b"\nendobj\n")

        xref_offset = f.tell()
        f.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            f.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        f.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )


def write_bar_chart_pdf(
    distribution: Counter[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = sorted(distribution.items())
    if not points:
        raise ValueError("No data to plot.")

    width = 432
    height = 288
    margin_left = 58
    margin_right = 18
    margin_top = 18
    margin_bottom = 52
    chart_left = margin_left
    chart_bottom = margin_bottom
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    max_count = max(count for _, count in points)
    tick_step = nice_tick_step(max_count, target_ticks=4)
    y_max = int(math.ceil(max_count / tick_step) * tick_step)
    y_max = max(y_max, tick_step)

    slot_width = chart_width / len(points)
    bar_width = min(slot_width * 0.56, 30)

    def x_center(i: int) -> float:
        return chart_left + slot_width * i + slot_width / 2

    def y_pos(value: int) -> float:
        return chart_bottom + chart_height * value / y_max

    commands: list[str] = [
        "q",
        set_fill("#ffffff"),
        f"0 0 {width} {height} re",
        "f",
    ]

    for tick in range(0, y_max + 1, tick_step):
        y = y_pos(tick)
        if tick:
            add_line(
                commands,
                chart_left,
                y,
                width - margin_right,
                y,
                color="#e3e8ef",
                width=0.45,
            )
        add_text(
            commands,
            str(tick),
            chart_left - 8,
            y - 3,
            font_size=8,
            color="#52606d",
            align="right",
        )

    add_line(
        commands,
        chart_left,
        chart_bottom,
        chart_left,
        chart_bottom + chart_height,
        color="#1f2933",
        width=0.75,
    )
    add_line(
        commands,
        chart_left,
        chart_bottom,
        width - margin_right,
        chart_bottom,
        color="#1f2933",
        width=0.75,
    )

    for i, (label_count, sample_count) in enumerate(points):
        center = x_center(i)
        bar_x = center - bar_width / 2
        bar_y = chart_bottom
        bar_height = y_pos(sample_count) - chart_bottom

        add_rect(
            commands,
            bar_x,
            bar_y,
            bar_width,
            bar_height,
            color="#4e79a7",
        )
        add_text(
            commands,
            str(sample_count),
            center,
            bar_y + bar_height + 5,
            font_size=8,
            color="#1f2933",
            bold=True,
            align="center",
        )
        add_text(
            commands,
            str(label_count),
            center,
            chart_bottom - 16,
            font_size=8.5,
            color="#52606d",
            align="center",
        )

    add_text(
        commands,
        "Number of labels per instance",
        chart_left + chart_width / 2,
        18,
        font_size=9.5,
        color="#1f2933",
        bold=True,
        align="center",
    )
    add_text(
        commands,
        "Number of instances",
        15,
        chart_bottom + chart_height / 2,
        font_size=9.5,
        color="#1f2933",
        bold=True,
        align="center",
        rotate_90=True,
    )
    commands.append("Q")

    write_pdf_page(output_path, commands, width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many unique labels each original sample has in "
            "processed_data_expanded.json and draw a bar chart."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset JSON path. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"PDF output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        help=f"Label field name. Default: {DEFAULT_LABEL_FIELD}",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help=(
            "Optional field used as original-sample id. If omitted, samples are grouped "
            "by all fields except ignored fields and the label field."
        ),
    )
    parser.add_argument(
        "--ignore-field",
        action="append",
        default=[],
        help=(
            "Extra field to ignore when reconstructing original samples. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset)
    ignored_fields = set(DEFAULT_IGNORED_FIELDS)
    ignored_fields.update(args.ignore_field)

    distribution, grouped_samples, _rows_without_label = count_labels_per_sample(
        samples,
        label_field=args.label_field,
        ignored_fields=ignored_fields,
        group_by=args.group_by,
    )

    write_bar_chart_pdf(
        distribution,
        args.output,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Grouped original instances: {grouped_samples}")
    print("Label-count distribution:")
    for label_count, sample_count in sorted(distribution.items()):
        print(f"  {label_count} label(s): {sample_count} instance(s)")
    print(f"Chart written to: {args.output}")


if __name__ == "__main__":
    main()
