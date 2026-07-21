#!/usr/bin/env python3
"""Build a complete, GitHub-sized ROMO-B manual from the scanned source PDF."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas


def page_number(path: Path) -> int:
    match = re.search(r"-(\d+)\.jpg$", path.name)
    if match is None:
        raise ValueError(f"Unexpected rendered page name: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--scale-to", type=int, default=2200)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--dpi", type=float, default=150.0)
    parser.add_argument("--expected-pages", type=int, default=38)
    parser.add_argument("--pdftoppm", type=Path)
    parser.add_argument(
        "--rotate-clockwise-pages",
        default="38",
        help="Comma-separated source page numbers whose scanned content is sideways",
    )
    args = parser.parse_args()

    input_pdf = args.input_pdf.expanduser().resolve()
    output_pdf = args.output_pdf.expanduser().resolve()
    if not input_pdf.is_file():
        raise FileNotFoundError(input_pdf)

    pdftoppm = str(args.pdftoppm) if args.pdftoppm else shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm was not found; install Poppler or pass --pdftoppm")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="romo_manual_pages_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [
                pdftoppm,
                "-jpeg",
                "-jpegopt",
                f"quality={args.jpeg_quality}",
                "-scale-to",
                str(args.scale_to),
                str(input_pdf),
                str(prefix),
            ],
            check=True,
        )
        pages = sorted(Path(tmp).glob("page-*.jpg"), key=page_number)
        if len(pages) != args.expected_pages:
            raise RuntimeError(f"Expected {args.expected_pages} pages, rendered {len(pages)}")

        rotate_pages = {
            int(value)
            for value in args.rotate_clockwise_pages.split(",")
            if value.strip()
        }

        pdf = canvas.Canvas(str(output_pdf), pageCompression=1)
        for page in pages:
            rendered_page = page
            if page_number(page) in rotate_pages:
                rendered_page = page.with_name(f"{page.stem}-clockwise.jpg")
                with Image.open(page) as image:
                    image.transpose(Image.Transpose.ROTATE_270).save(
                        rendered_page,
                        format="JPEG",
                        quality=args.jpeg_quality,
                        optimize=True,
                    )
            with Image.open(rendered_page) as image:
                width_px, height_px = image.size
            width_pt = width_px * 72.0 / args.dpi
            height_pt = height_px * 72.0 / args.dpi
            pdf.setPageSize((width_pt, height_pt))
            pdf.drawImage(
                str(rendered_page),
                0,
                0,
                width=width_pt,
                height=height_pt,
                preserveAspectRatio=True,
            )
            pdf.showPage()
        pdf.save()

    page_count = len(PdfReader(str(output_pdf)).pages)
    if page_count != args.expected_pages:
        raise RuntimeError(f"Output contains {page_count} pages")
    if output_pdf.stat().st_size >= 100_000_000:
        raise RuntimeError("Output exceeds GitHub's 100 MB single-file limit")

    print(f"created={output_pdf}")
    print(f"pages={page_count}")
    print(f"bytes={output_pdf.stat().st_size}")


if __name__ == "__main__":
    main()
