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
"""

import sys
import os
import shutil
import argparse
import subprocess
import time
import re
import zlib
import io
from pathlib import Path

try:
    import pikepdf
    from pikepdf import Stream, Name, Array, Dictionary
except Exception as e:
    print("Error: pikepdf required. Install with: pip3 install pikepdf", file=sys.stderr)
    raise

try:
    from PIL import Image
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
OP_TOKEN_RE = re.compile(rb'^[A-Za-z]{1,8}$')

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

# ---------- Binary Flate decode pass ----------
# This attempts to find stream objects whose dictionary contains "/Filter ... /FlateDecode"
# then replaces the bytes between "stream" and "endstream" with zlib.decompress(raw_bytes),
# and removes /Filter and /DecodeParms from the dictionary, updates /Length.
#
# This is a best-effort binary approach. It is intentionally conservative: if decompression
# fails we leave the original data untouched.
#
# Heuristics:
#  - Match: << ... /Filter ... >> <ws> stream <EOL> (data...) <EOL> endstream
#  - Support place of /Filter as single name or array: e.g. /Filter /FlateDecode or /Filter [ /FlateDecode /SomeOther ]
#  - Update the numeric /Length token in the dictionary to the new length
#
# NOTE: binary regexes on PDFs can fail for corner cases (nested dictionaries, v weird encodings).
# Use with caution.

# Pattern: find a dictionary followed by optional whitespace, `stream` token, newline, then data up to a newline+endstream
# We keep pattern fairly permissive: << ... >> may contain any bytes (non-greedy).
STREAM_OBJ_RE = re.compile(
    rb'(<<(?P<dict_body>.*?)>>)\s*stream\s*(?:\r\n|\n|\r)(?P<data>.*?)(?:\r\n|\n|\r)endstream',
    re.DOTALL
)

# match /Filter token referencing FlateDecode in the dictionary text
FILTER_FLATE_RE = re.compile(rb'/Filter\s*(\[(?P<fa>[^\]]*?)\]|(?P<fn>/(?:[^\s/>]+\b)))', re.DOTALL)

# Remove /Filter and /DecodeParms occurrences from the dictionary bytes
REMOVE_FILTER_DECODEPARMS_RE = re.compile(rb'(/\s*Filter\b\s*(?:\[[^\]]*?\]|/(?:[^\s/>]+))|/\s*DecodeParms\b\s*(?:\[[^\]]*?\]|\{[^\}]*\}|/(?:[^\s/>]+)|\([^)]*\))\s*)', re.DOTALL)

# Find and replace /Length <number>
LENGTH_RE = re.compile(rb'(/Length\s+)(\d+)', re.IGNORECASE)

def decode_flated_streams_in_pdf_bytes(pdf_bytes, logger=None):
    """
    Return (new_bytes, count_replaced) where new_bytes is modified binary PDF with any
    FlateDecode stream bodies replaced by their zlib-decompressed content (when decompression succeeds).
    """
    if logger is None:
        def logger(msg): pass

    out = bytearray()
    last_end = 0
    replaced = 0

    for m in STREAM_OBJ_RE.finditer(pdf_bytes):
        dict_bytes = m.group(1)   # includes the leading '<<' and trailing '>>'
        dict_body = m.group('dict_body') or b''
        stream_data = m.group('data') or b''
        start, end = m.span()
        # append preceding chunk
        out += pdf_bytes[last_end:start]

        # Check if this dictionary advertises FlateDecode in /Filter
        has_flated = False
        filt_match = FILTER_FLATE_RE.search(dict_bytes)
        if filt_match:
            # If there's a /Filter and it contains '/FlateDecode' text
            if b'/FlateDecode' in filt_match.group(0):
                has_flated = True

        if not has_flated:
            # copy original verbatim
            out += pdf_bytes[start:end]
            last_end = end
            continue

        # We have a /Filter referencing FlateDecode - attempt to zlib-decompress the stream data
        logger(f"[binary-flate-pass] Found candidate Flate stream at bytes {start}-{end}, trying zlib.decompress()")
        # The data captured here is the raw bytes between the newline after "stream" and the newline before "endstream".
        # It may include length bytes; try to decompress.
        try:
            # Attempt zlib raw decompress. Many PDF Flate streams are raw deflate with zlib wrapper.
            # Some streams are encoded in other ways (e.g. predictor params) and will fail here.
            decoded = zlib.decompress(stream_data)
        except Exception as e:
            # Try decompress with -zlib header (window size -zlib?), attempt with raw DEFLATE (wbits=-15)
            try:
                decoded = zlib.decompress(stream_data, wbits=-15)
            except Exception as e2:
                logger(f"[binary-flate-pass] zlib failed on region ({e}); leaving original compressed chunk")
                # fallback: keep original
                out += pdf_bytes[start:end]
                last_end = end
                continue

        # Build new dictionary bytes: remove /Filter and /DecodeParms entries
        dict_clean = REMOVE_FILTER_DECODEPARMS_RE.sub(b'', dict_bytes)
        # Update /Length number (add or replace). If no /Length present, we add it before the trailing >>.
        new_length = str(len(decoded)).encode('ascii')
        if LENGTH_RE.search(dict_clean):
            dict_clean = LENGTH_RE.sub(rb'\1' + new_length, dict_clean, count=1)
        else:
            # insert /Length N before the final '>>'
            # dict_bytes includes '<<' and '>>'. We must insert before >>; safe approach:
            # remove trailing '>>', strip trailing whitespace, append ' /Length <n> >>'
            inner = dict_clean
            # ensure we don't duplicate spaces awkwardly
            if inner.endswith(b' '):
                dict_clean = inner[:-1] + b' /Length ' + new_length + b' >>'
            else:
                dict_clean = inner + b' /Length ' + new_length + b' >>'

        # Compose replacement: dict_clean + newline + 'stream' + newline + decoded + newline + 'endstream'
        # Keep same EOL (prefer '\n')
        replacement = dict_clean + b'\nstream\n' + decoded + b'\nendstream'
        out += replacement
        replaced += 1
        last_end = end

    # append tail
    out += pdf_bytes[last_end:]
    return bytes(out), replaced

# ---------- helpers ----------
def tiny_png_bytes():
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

TINY_PNG = tiny_png_bytes()

def run_qpdf_uncompress(src_path: Path, dst_path: Path, logger):
    qpdf_path = shutil.which("qpdf")
    if not qpdf_path:
        logger(f"qpdf not found on PATH -> skipping decompression step for {src_path}")
        return False
    cmd = [qpdf_path, "--stream-data=uncompress", "--decode-level=all", "--object-streams=disable", str(src_path), str(dst_path)]
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

# ---------- content stream truncation ----------
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
        if len(out_chunks) == 0:
            trunc = stream_bytes[:max_ops * 16]
            logger("Truncation fallback: no operators detected; doing byte-level truncation")
            return trunc
        res = b'\n'.join(out_chunks)
        return res
    except Exception as e:
        logger(f"truncate_content_stream_bytes: exception {e}; falling back to raw truncation")
        return stream_bytes[:max_ops * 16]

def clear_object(obj):
    try:
        keys_to_delete = list(obj.keys())
        for k in keys_to_delete:
            if k == "/Length":
                continue
            try:
                del obj[k]
            except Exception:
                pass
    except Exception:
        pass

# ---------- main minimizer (pikepdf based) ----------
def minimize_pdf(in_pdf: Path, out_pdf: Path,
                 max_pages: int = DEFAULT_MAX_PAGES,
                 max_ops: int = DEFAULT_MAX_OPS,
                 image_threshold: int = DEFAULT_IMAGE_THRESHOLD,
                 logger=print):
    logger(f"[{now()}] Minimizing {in_pdf} -> {out_pdf} (max_pages={max_pages}, max_ops={max_ops}, image_thresh={image_threshold})")
    try:
        with pikepdf.Pdf.open(in_pdf, allow_overwriting_input=True) as pdf:
            total_pages = len(pdf.pages)
            if max_pages is not None and total_pages > max_pages:
                logger(f" - Trimming pages: {total_pages} -> {max_pages}")
                new_pdf = pikepdf.Pdf.new()
                for i in range(max_pages):
                    new_pdf.pages.append(pdf.pages[i])
                pdf.close()
                pdf = new_pdf

            replaced_images = 0
            # iterate pdf.objects directly
            for obj in pdf.objects:
                try:
                    # Try to decode Flate via pike/pdf interface in addition to our binary pass (harmless)
                    # but don't rely on it.
                    if isinstance(obj, pikepdf.Stream):
                        # If the object is an image stream, or anything large, check length keys first
                        subtype = obj.get("/Subtype")
                        if subtype == pikepdf.Name("/Image"):
                            # Try to read /Length without forcing decoding
                            try:
                                stream_len = int(obj.get("/Length") or 0)
                            except Exception:
                                try:
                                    # fallback to bytes length reading (may decode)
                                    stream_len = len(obj.read_bytes() or b'')
                                except Exception:
                                    stream_len = 0
                            if stream_len > image_threshold:
                                logger(f"   - Replacing large image object ({stream_len} bytes) with tiny PNG")
                                # clear dict (keep Length maybe)
                                keys_to_delete = list(obj.keys())
                                for k in keys_to_delete:
                                    try:
                                        del obj[k]
                                    except Exception:
                                        pass
                                obj["/Type"] = pikepdf.Name("/XObject")
                                obj["/Subtype"] = pikepdf.Name("/Image")
                                obj["/Width"] = 1
                                obj["/Height"] = 1
                                obj["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
                                obj["/BitsPerComponent"] = 8
                                obj.write(TINY_PNG)
                                replaced_images += 1
                except Exception as e:
                    logger(f"   - Warning while scanning objects: {e}")

            truncated_streams = 0
            for p_index, page in enumerate(pdf.pages):
                try:
                    contents = page.get("/Contents")
                    if contents is None:
                        continue
                    if isinstance(contents, pikepdf.Array):
                        new_contents = pikepdf.Array()
                        for c in contents:
                            try:
                                if isinstance(c, pikepdf.Stream):
                                    old_bytes = c.read_bytes() or b''
                                    new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                                    # wipe keys except /Length, then write new bytes
                                    keys_to_delete = list(c.keys())
                                    for k in keys_to_delete:
                                        if k != "/Length":
                                            try:
                                                del c[k]
                                            except Exception:
                                                pass
                                    c.write(new_bytes)
                                    c["/Length"] = len(new_bytes)
                                    new_contents.append(c)
                                    truncated_streams += 1
                                else:
                                    new_contents.append(c)
                            except Exception as e:
                                logger(f"     page {p_index}: error truncating stream array element: {e}")
                        page[pikepdf.Name("/Contents")] = new_contents
                    elif isinstance(contents, pikepdf.Stream):
                        old_bytes = contents.read_bytes() or b''
                        new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                        clear_object(contents)
                        contents.write(new_bytes)
                        contents["/Length"] = len(new_bytes)
                        truncated_streams += 1
                    else:
                        continue
                except Exception as e:
                    logger(f"  - Warning: while truncating content on page {p_index}: {e}")

            # Save; we disable recompress_flate and set stream_decode_level to none so we don't re-encode
            save_kwargs = {
                "linearize": True,
                "recompress_flate": False,
                "stream_decode_level": pikepdf.StreamDecodeLevel.none
            }
            tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
            pdf.save(str(tmp_out), **save_kwargs)
            tmp_out.replace(out_pdf)
            logger(f"Saved minimized PDF: {out_pdf} (replaced_images: {replaced_images}, truncated_streams: {truncated_streams})")
            return True, {"replaced_images": replaced_images, "truncated_streams": truncated_streams, "pages_kept": len(pdf.pages)}
    except Exception as e:
        logger(f"Minimizer error for {in_pdf}: {e}")
        return False, {"error": str(e)}

# ---------- top-level ----------
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
        dec_out = dec_dir / f.name
        ok_qpdf = run_qpdf_uncompress(f, dec_out, logger)
        if not ok_qpdf:
            logger(" - qpdf failed or missing -> copying original as decompressed fallback")
            safe_copy(f, dec_out, logger)
        else:
            # After qpdf produced dec_out, do binary Flate->raw substitution pass on dec_out.
            try:
                data = dec_out.read_bytes()
                new_data, count = decode_flated_streams_in_pdf_bytes(data, logger=logger)
                if count > 0:
                    # Backup original qpdf output before overwriting
                    bak = dec_out.with_suffix(dec_out.suffix + ".qpdfbak")
                    try:
                        dec_out.replace(bak)
                        # write new_data to dec_out (do atomic replace)
                        tmp = dec_out.with_suffix(dec_out.suffix + ".tmpbin")
                        tmp.write_bytes(new_data)
                        tmp.replace(dec_out)
                        logger(f" - Binary flate decode pass replaced {count} streams in {dec_out.name}")
                    except Exception as e:
                        logger(f" - Failed to atomically write flate-decoded PDF: {e}")
                        # try fallback: write to backup file instead
                        try:
                            dec_out.write_bytes(new_data)
                            logger(" - Wrote flate-decoded file (fallback)")
                        except Exception as e2:
                            logger(f" - Failed to write flate-decoded file: {e2}")
                else:
                    logger(" - Binary flate pass made no replacements")
            except Exception as e:
                logger(f" - Binary flate pass failure: {e}")

        # now minimize
        min_out = min_dir / f.name
        succ, info = minimize_pdf(dec_out, min_out,
                                  max_pages=args.max_pages,
                                  max_ops=args.max_ops,
                                  image_threshold=args.image_threshold,
                                  logger=logger)
        if not succ:
            logger(f"Minimization failed for {f.name} !!!")
            try:
                # copy failing file for triage
                failed_path = failed_dir / f.name
                safe_copy(dec_out, failed_path, logger)
            except Exception:
                pass

    logger("Done.")

if __name__ == "__main__":
    main()