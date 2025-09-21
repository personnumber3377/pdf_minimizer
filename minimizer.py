#!/usr/bin/env python3
"""
Decompress and minimize a PDF corpus, with optional parallel workers.

Usage:
    decompress_and_minimize.py INPUT_DIR DECOMPRESSED_DIR MINIMIZED_DIR [--workers N]
"""

import sys
import os
import shutil
import argparse
import subprocess
import time
import re
import zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pikepdf
    from pikepdf import Stream, Name, Array, Dictionary
except Exception as e:
    print("Error: pikepdf required. Install with: pip3 install pikepdf", file=sys.stderr)
    raise

try:
    from PIL import Image
    import io
except Exception as e:
    print("Error: Pillow required. Install with: pip3 install Pillow", file=sys.stderr)
    raise


DEFAULT_MAX_PAGES = 5000
DEFAULT_MAX_OPS = 400000
DEFAULT_IMAGE_THRESHOLD = 100 * 1024
LOG_ENCODING = "utf-8"

OP_TOKEN_RE = re.compile(rb'^[A-Za-z]{1,8}$')


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def tiny_png_bytes():
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


TINY_PNG = tiny_png_bytes()


def run_qpdf_uncompress(src_path: Path, dst_path: Path):
    qpdf_path = shutil.which("qpdf")
    if not qpdf_path:
        return False, f"qpdf not found"
    cmd = [
        qpdf_path,
        "--stream-data=uncompress",
        "--decode-level=all",
        "--object-streams=disable",
        str(src_path),
        str(dst_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, f"qpdf ok"
    except subprocess.CalledProcessError as e:
        return False, f"qpdf failed: {e.stderr.decode(errors='ignore')}"


def safe_copy(src: Path, dst: Path):
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        return f"copy failed: {e}"
    return None


def is_pdf_filename(p: Path):
    return p.suffix.lower() == ".pdf"


def raw_decompress_flate_streams(pdf_path: Path):
    data = pdf_path.read_bytes()
    out = bytearray()
    pos = 0
    pattern = re.compile(rb"(/Filter\s*/FlateDecode[^\n\r]*?)stream[\r\n]+", re.S)

    while True:
        m = pattern.search(data, pos)
        if not m:
            out.extend(data[pos:])
            break
        start, end = m.span()
        out.extend(data[pos:start])
        dict_part = m.group(1)
        stream_start = end
        stream_end = data.find(b"endstream", stream_start)
        if stream_end == -1:
            out.extend(data[start:])
            break
        compressed = data[stream_start:stream_end].strip(b"\r\n")
        try:
            decompressed = zlib.decompress(compressed)
            new_dict = re.sub(rb"/Filter\s*/FlateDecode", b"", dict_part)
            new_dict = re.sub(rb"/Length\s+\d+", f"/Length {len(decompressed)}".encode(), new_dict)
            out.extend(new_dict)
            out.extend(b"\nstream\n")
            out.extend(decompressed)
            out.extend(b"\nendstream")
        except Exception:
            out.extend(data[start:stream_end + len(b"endstream")])
        pos = stream_end + len(b"endstream")
    pdf_path.write_bytes(out)


def truncate_content_stream_bytes(stream_bytes: bytes, max_ops: int) -> bytes:
    try:
        tokens = re.split(rb'(\s+)', stream_bytes)
        out_chunks = []
        operand_buffer = []
        op_count = 0
        flat_tokens = [t for t in tokens if t != b'']
        i = 0
        while i < len(flat_tokens):
            t = flat_tokens[i]
            if t.isspace():
                if operand_buffer:
                    operand_buffer.append(b' ')
                i += 1
                continue
            if OP_TOKEN_RE.match(t):
                operand_buffer.append(t)
                out_chunks.append(b''.join(operand_buffer))
                operand_buffer = []
                op_count += 1
                if op_count >= max_ops:
                    break
            else:
                operand_buffer.append(t)
                operand_buffer.append(b' ')
            i += 1
        if not out_chunks:
            return stream_bytes[:max_ops * 16]
        return b'\n'.join(out_chunks)
    except Exception:
        return stream_bytes[:max_ops * 16]


def clear_object(obj):
    for k in list(obj.keys()):
        if k != "/Length":
            del obj[k]


def minimize_pdf(in_pdf: Path, out_pdf: Path, max_pages, max_ops, image_threshold):
    with pikepdf.Pdf.open(in_pdf, allow_overwriting_input=True) as pdf:
        if max_pages and len(pdf.pages) > max_pages:
            new_pdf = pikepdf.Pdf.new()
            for i in range(max_pages):
                new_pdf.pages.append(pdf.pages[i])
            pdf.close()
            pdf = new_pdf
        for obj in pdf.objects:
            if not isinstance(obj, pikepdf.Stream):
                continue
            subtype = obj.get("/Subtype")
            if subtype != pikepdf.Name("/Image"):
                continue
            stream_len = obj.get("/Length", 0)
            if isinstance(stream_len, int) and stream_len > image_threshold:
                for k in list(obj.keys()):
                    if k != "/Length":
                        del obj[k]
                obj["/Type"] = pikepdf.Name("/XObject")
                obj["/Subtype"] = pikepdf.Name("/Image")
                obj["/Width"] = 1
                obj["/Height"] = 1
                obj["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
                obj["/BitsPerComponent"] = 8
                obj.write(TINY_PNG)
        for page in pdf.pages:
            contents = page.get("/Contents")
            if contents is None:
                continue
            if isinstance(contents, pikepdf.Array):
                new_contents = pikepdf.Array()
                for c in contents:
                    if isinstance(c, pikepdf.Stream):
                        new_bytes = truncate_content_stream_bytes(c.read_bytes() or b'', max_ops)
                        clear_object(c)
                        c.write(new_bytes)
                    new_contents.append(c)
                page[pikepdf.Name("/Contents")] = new_contents
            elif isinstance(contents, pikepdf.Stream):
                new_bytes = truncate_content_stream_bytes(contents.read_bytes() or b'', max_ops)
                clear_object(contents)
                contents.write(new_bytes)
        tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
        pdf.save(str(tmp_out), linearize=True, compress_streams=False)
        tmp_out.replace(out_pdf)


def process_file(f: Path, dec_dir: Path, min_dir: Path, args) -> str:
    dec_out = dec_dir / f.name
    ok, msg = run_qpdf_uncompress(f, dec_out)
    if not ok:
        safe_copy(f, dec_out)
    raw_decompress_flate_streams(dec_out)
    min_out = min_dir / f.name
    minimize_pdf(dec_out, min_out, args.max_pages, args.max_ops, args.image_threshold)
    return f"Processed {f.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("decompressed_dir")
    ap.add_argument("minimized_dir")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--max-ops", type=int, default=DEFAULT_MAX_OPS)
    ap.add_argument("--image-threshold", type=int, default=DEFAULT_IMAGE_THRESHOLD)
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    dec_dir = Path(args.decompressed_dir)
    min_dir = Path(args.minimized_dir)

    dec_dir.mkdir(parents=True, exist_ok=True)
    min_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.iterdir() if p.is_file() and is_pdf_filename(p)]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, f, dec_dir, min_dir, args): f for f in files}
            for fut in as_completed(futures):
                try:
                    print(fut.result())
                except Exception as e:
                    print(f"Error on {futures[fut].name}: {e}")
    else:
        for f in files:
            try:
                print(process_file(f, dec_dir, min_dir, args))
            except Exception as e:
                print(f"Error on {f.name}: {e}")


if __name__ == "__main__":
    main()