#!/usr/bin/env python3
"""
Decompress and minimize a PDF corpus.

Usage:
    decompress_and_minimize.py INPUT_DIR DECOMPRESSED_DIR MINIMIZED_DIR
Options (env/flags):
    --max-pages N         Limit pages kept per PDF (default 5)
    --max-ops N           Max PDF operators to keep per content stream (default 400)
    --image-threshold K   Replace images larger than K bytes (default 100*1024)
    --log FILE            Append log to FILE (default ./minimizer.log)
    --workers N           Not implemented (single-threaded); placeholder
"""

import sys
import os
import shutil
import argparse
import subprocess
import tempfile
import hashlib
import time
import re
from pathlib import Path

try:
    import pikepdf
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
DEFAULT_MAX_PAGES = 5
DEFAULT_MAX_OPS = 400
DEFAULT_IMAGE_THRESHOLD = 100 * 1024   # bytes
LOG_ENCODING = "utf-8"

# Regex to identify PDF operator tokens (simple heuristic)
# Operators are typically alphabetic tokens like 'Tf', 'Tj', 'm', 'l', 're', 'S', 'f', 'cm', ...
OP_TOKEN_RE = re.compile(rb'^[A-Za-z]{1,8}$')

# -----------------------------
# Helpers
# -----------------------------
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def tiny_png_bytes():
    """Return bytes for a 1x1 white PNG as a replacement image."""
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

TINY_PNG = tiny_png_bytes()

def run_qpdf_uncompress(src_path: Path, dst_path: Path, logger):
    """Attempt to run qpdf --stream-data=uncompress src dst.
       Returns True if successful, False otherwise (in which case caller should fallback)."""
    qpdf_path = shutil.which("qpdf")
    if not qpdf_path:
        logger(f"qpdf not found on PATH -> skipping decompression step for {src_path}")
        return False
    cmd = [qpdf_path, "--stream-data=uncompress", str(src_path), str(dst_path)]
    logger(f"Running qpdf to uncompress streams: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger(f"qpdf failed on {src_path} (exit {e.returncode}), stderr: {e.stderr.decode(errors='ignore')}")
        return False
    except Exception as e:
        logger(f"qpdf run error: {e}")
        return False

def safe_copy(src: Path, dst: Path, logger):
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logger(f"failed to copy {src} -> {dst}: {e}")

def is_pdf_filename(p: Path):
    return p.suffix.lower() == ".pdf"

# -----------------------------
# Content stream truncation
# -----------------------------
def truncate_content_stream_bytes(stream_bytes: bytes, max_ops: int, logger) -> bytes:
    """
    Heuristic operator-aware truncation:
    - Split stream into whitespace tokens.
    - Recognize operator tokens using OP_TOKEN_RE.
    - Emit sequences (operands... operator) until max_ops operators reached.
    - If parsing fails or no operators found, fallback to byte-truncate.
    """
    try:
        tokens = re.split(rb'(\s+)', stream_bytes)  # keep whitespace tokens in list
        # We'll walk tokens and assemble operand lists; operators are tokens matching RE.
        out_chunks = []
        operand_buffer = []
        op_count = 0
        # iterate over tokens but skip raw whitespace tokens for identification
        flat_tokens = [t for t in tokens if t != b'']  # includes whitespace pieces
        # For easier parse, collapse whitespace into single spaces in reconstruction where needed
        i = 0
        while i < len(flat_tokens):
            t = flat_tokens[i]
            if t.isspace():
                # whitespace - append as-is to buffer if we hold any operand
                if operand_buffer:
                    operand_buffer.append(b' ')
                i += 1
                continue
            # if token is an operator
            if OP_TOKEN_RE.match(t):
                # treat token as operator
                operand_buffer.append(t)
                # flush buffer to out
                out_chunks.append(b''.join(operand_buffer))
                operand_buffer = []
                op_count += 1
                if op_count >= max_ops:
                    break
            else:
                # operand token - add to operand buffer with trailing space
                operand_buffer.append(t)
                operand_buffer.append(b' ')
            i += 1

        if len(out_chunks) == 0:
            # No operators recognized - fallback to raw truncation
            trunc = stream_bytes[:max_ops * 16]
            logger("Truncation fallback: no operators detected; doing byte-level truncation")
            return trunc
        # join output chunks using single newline between them
        res = b'\n'.join(out_chunks)
        return res
    except Exception as e:
        logger(f"truncate_content_stream_bytes: exception {e}; falling back to raw truncation")
        return stream_bytes[:max_ops * 16]

# -----------------------------
# Main PDF minimizer
# -----------------------------
def minimize_pdf(in_pdf: Path, out_pdf: Path,
                 max_pages: int = DEFAULT_MAX_PAGES,
                 max_ops: int = DEFAULT_MAX_OPS,
                 image_threshold: int = DEFAULT_IMAGE_THRESHOLD,
                 logger=print):
    """
    Open PDF with pikepdf, perform safe heuristics:
    - Optionally limit pages to first max_pages
    - For each page content stream: truncate by operators (max_ops)
    - Replace images whose stream length > image_threshold with 1x1 PNG
    - Save cleaned file (pikepdf will rebuild xref so unreachable objects are dropped)
    """
    logger(f"[{now()}] Minimizing {in_pdf} -> {out_pdf} (max_pages={max_pages}, max_ops={max_ops}, image_thresh={image_threshold})")
    try:
        with pikepdf.Pdf.open(in_pdf, allow_overwriting_input=True) as pdf:
            # Limit pages if requested
            total_pages = len(pdf.pages)
            if max_pages is not None and total_pages > max_pages:
                logger(f" - Trimming pages: {total_pages} -> {max_pages}")
                # keep first max_pages pages
                # Rebuild pdf with only first pages
                new_pdf = pikepdf.Pdf.new()
                for i in range(max_pages):
                    new_pdf.pages.append(pdf.pages[i])
                pdf.close()
                pdf = new_pdf

            # Replace large images
            replaced_images = 0
            # iterate all objects: pikepdf makes them dict-like; use list(pdf.objects) to avoid mutating view
            for objnum, obj in list(pdf.objects.items()):
                try:
                    # only consider stream objects with /Subtype /Image
                    if not isinstance(obj, pikepdf.Stream):
                        continue
                    # many image XObjects have Subtype /Image
                    subtype = obj.get("/Subtype")
                    if subtype != pikepdf.Name("/Image"):
                        continue
                    # stream size check
                    stream_len = len(obj.read_bytes() or b'')
                    if stream_len > image_threshold:
                        # replace with small PNG stream
                        logger(f"   - Replacing large image object {objnum} ({stream_len} bytes) with tiny PNG")
                        # Build a minimal image dict that looks like an XObject image but with PNG stream
                        # Simpler: set stream bytes to tiny PNG and remove/modify potentially incompatible keys.
                        # Keep BitsPerComponent and ColorSpace if they exist and are simple Names.
                        newdict = pikepdf.Dictionary()
                        # Use XObject via Type/Subtype
                        newdict["/Type"] = pikepdf.Name("/XObject")
                        newdict["/Subtype"] = pikepdf.Name("/Image")
                        # Keep ColorSpace if simple name
                        if "/ColorSpace" in obj:
                            try:
                                cs = obj["/ColorSpace"]
                                if isinstance(cs, pikepdf.Name):
                                    newdict["/ColorSpace"] = cs
                            except Exception:
                                pass
                        if "/BitsPerComponent" in obj:
                            try:
                                bpc = obj["/BitsPerComponent"]
                                if isinstance(bpc, int):
                                    newdict["/BitsPerComponent"] = bpc
                            except Exception:
                                pass
                        # replace the stream
                        obj.clear()
                        for k, v in newdict.items():
                            obj[k] = v
                        obj.write_bytes(TINY_PNG)
                        replaced_images += 1
                except Exception as e:
                    logger(f"   - Warning: while scanning image object {objnum}: {e}")

            # Truncate content streams for each page
            truncated_streams = 0
            for p_index, page in enumerate(pdf.pages):
                try:
                    contents = page.get("/Contents")
                    if contents is None:
                        continue
                    # Contents can be single stream or array of streams
                    if isinstance(contents, pikepdf.Array):
                        new_contents = pikepdf.Array()
                        for c in contents:
                            try:
                                if isinstance(c, pikepdf.Stream):
                                    old_bytes = c.read_bytes() or b''
                                    new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                                    c.clear()
                                    c.write_bytes(new_bytes)
                                    new_contents.append(c)
                                    truncated_streams += 1
                                else:
                                    new_contents.append(c)
                            except Exception as e:
                                logger(f"     page {p_index}: error truncating stream array element: {e}")
                        page[ pikepdf.Name("/Contents") ] = new_contents
                    elif isinstance(contents, pikepdf.Stream):
                        old_bytes = contents.read_bytes() or b''
                        new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                        contents.clear()
                        contents.write_bytes(new_bytes)
                        truncated_streams += 1
                    else:
                        # weird type
                        continue
                except Exception as e:
                    logger(f"  - Warning: while truncating content on page {p_index}: {e}")

            # Save minimized PDF (let pikepdf rebuild xref and drop unreferenced objects)
            save_kwargs = {"linearize": True}
            # write to temp and then move to avoid partial writes
            tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
            pdf.save(str(tmp_out), **save_kwargs)
            tmp_out.replace(out_pdf)
            logger(f"Saved minimized PDF: {out_pdf} (replaced images: {replaced_images}, truncated streams: {truncated_streams})")
            return True, {"replaced_images": replaced_images, "truncated_streams": truncated_streams, "pages_kept": len(pdf.pages)}
    except Exception as e:
        logger(f"Minimizer error for {in_pdf}: {e}")
        return False, {"error": str(e)}

# -----------------------------
# Top-level flow
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Decompress and minimize PDF corpus")
    ap.add_argument("input_dir", help="Directory with original PDFs")
    ap.add_argument("decompressed_dir", help="Directory to write decompressed PDFs")
    ap.add_argument("minimized_dir", help="Directory to write minimized PDFs")
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
    failed_dir = min_dir / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)

    def logger(msg):
        ts = now()
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(log_file, "a", encoding=LOG_ENCODING) as lf:
                lf.write(line + "\n")
        except Exception:
            pass

    logger(f"Starting decompress+minimize run. input={input_dir}, decompressed={dec_dir}, minimized={min_dir}")
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and is_pdf_filename(p)])
    if not files:
        logger("No .pdf files found in input dir; exiting.")
        sys.exit(0)

    for f in files:
        logger(f"Processing {f.name}")
        # create decompressed path
        dec_out = dec_dir / f.name
        # attempt qpdf decompression
        ok_qpdf = run_qpdf_uncompress(f, dec_out, logger)
        if not ok_qpdf:
            # fallback: copy original to decompressed dir
            logger(" - qpdf failed or missing -> copying original as decompressed fallback")
            safe_copy(f, dec_out, logger)
        # now minimize
        min_out = min_dir / f.name
        succ, info = minimize_pdf(dec_out, min_out,
                                  max_pages=args.max_pages,
                                  max_ops=args.max_ops,
                                  image_threshold=args.image_threshold,
                                  logger=logger)
        if not succ:
            # save failing file into failed dir for manual triage
            fallback_path = failed_dir / (f.name + ".bad")
            logger(f" - Minimization failed for {f.name}; copying original to {fallback_path}")
            try:
                safe_copy(dec_out, fallback_path, logger)
            except Exception as e:
                logger(f" - failed to copy fallback: {e}")

    logger("Done.")

if __name__ == "__main__":
    main()
