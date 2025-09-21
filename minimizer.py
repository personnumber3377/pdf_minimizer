#!/usr/bin/env python3
"""
Decompress and minimize a PDF corpus.

Usage:
    decompress_and_minimize.py INPUT_DIR DECOMPRESSED_DIR MINIMIZED_DIR
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


# -----------------------------
# Configurable defaults
# -----------------------------
# DEFAULT_MAX_PAGES = 5
# DEFAULT_MAX_OPS = 400

DEFAULT_MAX_PAGES = 5000
DEFAULT_MAX_OPS = 400000

DEFAULT_IMAGE_THRESHOLD = 100 * 1024   # bytes
LOG_ENCODING = "utf-8"

OP_TOKEN_RE = re.compile(rb'^[A-Za-z]{1,8}$')


# -----------------------------
# Helpers
# -----------------------------
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def tiny_png_bytes():
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


TINY_PNG = tiny_png_bytes()


def run_qpdf_uncompress(src_path: Path, dst_path: Path, logger):
    qpdf_path = shutil.which("qpdf")
    if not qpdf_path:
        logger(f"qpdf not found -> skipping decompression for {src_path}")
        return False
    cmd = [
        qpdf_path,
        "--stream-data=uncompress",
        "--decode-level=all",
        "--object-streams=disable",
        str(src_path),
        str(dst_path),
    ]
    logger(f"Running qpdf: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger(f"qpdf failed on {src_path} (exit {e.returncode}): {e.stderr.decode(errors='ignore')}")
        return False


def safe_copy(src: Path, dst: Path, logger):
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logger(f"failed to copy {src} -> {dst}: {e}")


def is_pdf_filename(p: Path):
    return p.suffix.lower() == ".pdf"


# -----------------------------
# Raw FlateDecode decompressor
# -----------------------------
def raw_decompress_flate_streams(pdf_path: Path, logger):
    """
    Parse raw PDF bytes, find `/Filter /FlateDecode ... stream ... endstream`,
    zlib-decompress content, and replace it with uncompressed content.
    Writes result back to same path.
    """
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
            logger(f"[raw] no endstream after offset {stream_start}, stopping")
            out.extend(data[start:])
            break

        compressed = data[stream_start:stream_end].strip(b"\r\n")

        try:
            decompressed = zlib.decompress(compressed)
            logger(f"[raw] decompressed Flate stream: {len(compressed)} -> {len(decompressed)}")
            # remove /Filter entry from dictionary
            new_dict = re.sub(rb"/Filter\s*/FlateDecode", b"", dict_part)
            # update /Length
            new_dict = re.sub(rb"/Length\s+\d+", f"/Length {len(decompressed)}".encode(), new_dict)
            out.extend(new_dict)
            out.extend(b"\nstream\n")
            out.extend(decompressed)
            out.extend(b"\nendstream")
        except Exception as e:
            logger(f"[raw] zlib failed: {e}; keeping compressed")
            out.extend(data[start:stream_end + len(b"endstream")])

        pos = stream_end + len(b"endstream")

    pdf_path.write_bytes(out)


# -----------------------------
# Content stream truncation
# -----------------------------
def truncate_content_stream_bytes(stream_bytes: bytes, max_ops: int, logger) -> bytes:
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
            trunc = stream_bytes[:max_ops * 16]
            logger("Truncation fallback: byte-level")
            return trunc
        return b'\n'.join(out_chunks)
    except Exception as e:
        logger(f"truncate_content_stream_bytes error: {e}")
        return stream_bytes[:max_ops * 16]


def clear_object(obj):
    keys_to_delete = list(obj.keys())
    for k in keys_to_delete:
        if k == "/Length":
            continue
        del obj[k]


# -----------------------------
# Main PDF minimizer
# -----------------------------
def minimize_pdf(in_pdf: Path, out_pdf: Path,
                 max_pages: int,
                 max_ops: int,
                 image_threshold: int,
                 logger=print):
    logger(f"[{now()}] Minimizing {in_pdf} -> {out_pdf}")
    try:
        with pikepdf.Pdf.open(in_pdf, allow_overwriting_input=True) as pdf:
            total_pages = len(pdf.pages)
            if max_pages is not None and total_pages > max_pages:
                new_pdf = pikepdf.Pdf.new()
                for i in range(max_pages):
                    new_pdf.pages.append(pdf.pages[i])
                pdf.close()
                pdf = new_pdf

            replaced_images = 0
            for obj in pdf.objects:
                if not isinstance(obj, pikepdf.Stream):
                    continue
                subtype = obj.get("/Subtype")
                if subtype != pikepdf.Name("/Image"):
                    continue
                stream_len = obj.get("/Length", 0)
                if isinstance(stream_len, int) and stream_len > image_threshold:
                    logger(f"   - Replacing image {stream_len} bytes with tiny PNG")
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
                    replaced_images += 1

            truncated_streams = 0
            for p_index, page in enumerate(pdf.pages):
                contents = page.get("/Contents")
                if contents is None:
                    continue
                if isinstance(contents, pikepdf.Array):
                    new_contents = pikepdf.Array()
                    for c in contents:
                        if isinstance(c, pikepdf.Stream):
                            old_bytes = c.read_bytes() or b''
                            new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                            clear_object(c)
                            c.write(new_bytes)
                            truncated_streams += 1
                        new_contents.append(c)
                    page[pikepdf.Name("/Contents")] = new_contents
                elif isinstance(contents, pikepdf.Stream):
                    old_bytes = contents.read_bytes() or b''
                    new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                    clear_object(contents)
                    contents.write(new_bytes)
                    truncated_streams += 1

            tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
            pdf.save(str(tmp_out), linearize=True, compress_streams=False)
            tmp_out.replace(out_pdf)

            logger(f"Saved minimized {out_pdf}")
            return True, {"replaced_images": replaced_images,
                          "truncated_streams": truncated_streams,
                          "pages_kept": len(pdf.pages)}
    except:
        pass


# -----------------------------
# Top-level flow
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("decompressed_dir")
    ap.add_argument("minimized_dir")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--max-ops", type=int, default=DEFAULT_MAX_OPS)
    ap.add_argument("--image-threshold", type=int, default=DEFAULT_IMAGE_THRESHOLD)
    ap.add_argument("--log", default="minimizer.log")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    dec_dir = Path(args.decompressed_dir)
    min_dir = Path(args.minimized_dir)
    log_file = Path(args.log)

    dec_dir.mkdir(parents=True, exist_ok=True)
    min_dir.mkdir(parents=True, exist_ok=True)

    def logger(msg):
        ts = now()
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(log_file, "a", encoding=LOG_ENCODING) as lf:
                lf.write(line + "\n")
        except Exception:
            pass

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and is_pdf_filename(p)])
    for f in files:
        logger(f"Processing {f.name}")
        dec_out = dec_dir / f.name
        if not run_qpdf_uncompress(f, dec_out, logger):
            safe_copy(f, dec_out, logger)
        # force raw FlateDecode decompression
        raw_decompress_flate_streams(dec_out, logger)
        min_out = min_dir / f.name
        minimize_pdf(dec_out, min_out,
                     max_pages=args.max_pages,
                     max_ops=args.max_ops,
                     image_threshold=args.image_threshold,
                     logger=logger)

    logger("Done.")


if __name__ == "__main__":
    main()