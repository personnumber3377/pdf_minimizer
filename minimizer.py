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
import zlib
from pathlib import Path

try:
    import pikepdf
    # from pikepdf import Name, Stream, Array, Dictionary
    from pikepdf import Stream, Name, Array, Dictionary, StreamDecodeLevel
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

'''
def decode_flate_stream(obj, logger):
    logger(f"Called with {obj}")
    try:
        # if not isinstance(obj, pikepdf.Stream):
        #     return False
        filt = obj.get("/Filter")
        if filt == pikepdf.Name("/FlateDecode") or (
            isinstance(filt, pikepdf.Array) and pikepdf.Name("/FlateDecode") in filt
        ):
            print("Has filter!!!!")
            raw = obj.read_raw_bytes()   # get compressed bytes without decoding
            logger(f"Raw: {raw}")
            try:
                decoded = zlib.decompress(raw)
            except Exception as e:
                logger(f"   - Failed to decompress FlateDecode stream: {e}")
                return False

            # Clear dict entries that point to filters
            keys_to_delete = [k for k in obj.keys() if k in ("/Filter", "/DecodeParms")]
            for k in keys_to_delete:
                try:
                    del obj[k]
                except Exception:
                    pass

            obj.write(decoded)  # overwrite with decoded content
            obj["/Length"] = len(decoded)
            logger(f"   - Replaced FlateDecode stream with raw decoded ({len(decoded)} bytes)")
            return True
        return False
    except Exception as e:
        logger(f"   - Error in decode_flate_stream: {e}")
        return False
'''

'''
def decode_flate_stream(obj, logger, seen=None, path="root"):
    """
    Recursively search for and decode FlateDecode streams inside obj.
    Replaces compressed data with decoded bytes, strips /Filter & /DecodeParms.
    """
    if seen is None:
        seen = set()
    # Avoid infinite recursion on the same object
    try:
        objid = getattr(obj, "objgen", None)
        if objid is not None and objid in seen:
            return False
        if objid is not None:
            seen.add(objid)
    except Exception:
        # print("Fuck!!!")
        pass

    decoded_any = False

    try:
        # Handle if it's a Stream object
        if isinstance(obj, Stream):
            print("Is stream!!!")
            filt = obj.get("/Filter")
            if filt == Name("/FlateDecode") or (
                isinstance(filt, Array) and Name("/FlateDecode") in filt
            ):
                logger(f"[{path}] Found FlateDecode stream in {obj}")
                try:
                    raw = obj.read_raw_bytes()   # compressed data
                except Exception as e:
                    logger(f"[{path}] Failed to read raw bytes: {e}")
                    raw = None

                if raw:
                    try:
                        decoded = zlib.decompress(raw)
                        # Drop filter keys
                        for k in ["/Filter", "/DecodeParms"]:
                            if k in obj:
                                del obj[k]
                        obj.write(decoded)
                        obj["/Length"] = len(decoded)
                        logger(f"[{path}] Decoded Flate stream -> {len(decoded)} bytes")
                        decoded_any = True
                    except Exception as e:
                        logger(f"[{path}] zlib failed: {e}")

        # Recurse if it's a dictionary-like object
        if hasattr(obj, "items"):
            print("current obj.items: "+str(obj.items)) # Something like this???
            print("obj: "+str(obj))
            if isinstance(obj, pikepdf.Array):
                # Array...
                print("obj: "+str(obj)+" is an array!!!!!!")
                # isinstance(obj, Array):
                for idx, v in enumerate(obj):
                    if isinstance(v, (Stream, Dictionary, Array)):
                        if decode_flate_stream(v, logger, seen, path=f"{path}[{idx}]"):
                            decoded_any = True
                return decoded_any

            for k, v in obj.items():
                if isinstance(v, (Stream, Dictionary, Array)):
                    if decode_flate_stream(v, logger, seen, path=f"{path}/{k}"):
                        decoded_any = True

        # Recurse if it's an array-like object
        elif isinstance(obj, Array):
            for idx, v in enumerate(obj):
                if isinstance(v, (Stream, Dictionary, Array)):
                    if decode_flate_stream(v, logger, seen, path=f"{path}[{idx}]"):
                        decoded_any = True

    except Exception as e:
        logger(f"[{path}] Error: {e}")

    return decoded_any
'''


def decode_flate_stream(obj, logger, seen=None, path="root"):
    """
    Recursively traverse and ensure FlateDecode streams are fully decoded.
    Works even if qpdf has already decoded at open time.
    """
    if seen is None:
        seen = set()
    try:
        objid = getattr(obj, "objgen", None)
        if objid is not None and objid in seen:
            return False
        if objid is not None:
            seen.add(objid)
    except Exception:
        pass

    decoded_any = False

    try:
        if isinstance(obj, Stream):
            print(obj.__dir__())
            # Explicitly request raw vs. decoded
            try:
                raw = obj.read_raw_bytes()     # compressed (if still available)
                decoded = obj.read_bytes(StreamDecodeLevel.generalized)  # decoded form
            except Exception as e:
                logger(f"[{path}] cannot read stream: {e}")
                return False
            print("raw: "+str(raw))
            # If Filter entry is gone, assume it was already decoded
            # filt = obj.get("/Filter")
            # if filt is not None:
            # logger(f"[{path}] stream still has Filter={filt}, forcing decode")

            # try:
            #     decoded2 = zlib.decompress(raw)
            # except Exception as e:
            #     logger(f"[{path}] zlib failed: {e}")
            #     decoded2 = decoded  # fallback to pikepdf’s decode
            decoded2 = raw
            # clear filter keys
            # for k in ["/Filter", "/DecodeParms"]:
            #     if k in obj:
            #         del obj[k]

            obj.write(decoded2)
            # obj["/Length"] = len(decoded2)
            logger(f"[{path}] replaced with uncompressed stream, {len(decoded2)} bytes")
            decoded_any = True
            obj = None # Set to none...



            #else:
            #    # Already decoded by pikepdf
            #    logger(f"[{path}] stream already decoded ({len(decoded)} bytes)")
        
        # Recurse into dicts/arrays
        if hasattr(obj, "items"):

            if isinstance(obj, pikepdf.Array):
                # Array...
                print("obj: "+str(obj)+" is an array!!!!!!")
                # isinstance(obj, Array):
                for idx, v in enumerate(obj):
                    if isinstance(v, (Stream, Dictionary, Array)):
                        if decode_flate_stream(v, logger, seen, path=f"{path}[{idx}]"):
                            decoded_any = True
                return decoded_any

            for k, v in obj.items():
                if isinstance(v, (Stream, Dictionary, Array)):
                    if decode_flate_stream(v, logger, seen, f"{path}/{k}"):
                        decoded_any = True

        elif isinstance(obj, Array):
            for idx, v in enumerate(obj):
                if isinstance(v, (Stream, Dictionary, Array)):
                    if decode_flate_stream(v, logger, seen, f"{path}[{idx}]"):
                        decoded_any = True

    except Exception as e:
        logger(f"[{path}] Error: {e}")

    return decoded_any


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
    cmd = [qpdf_path, "--stream-data=uncompress", "--decode-level=all", "--object-streams=disable", str(src_path), str(dst_path)] # Uncompress all shit...
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
        raise(e)
        return stream_bytes[:max_ops * 16]

def clear_object(obj): # Clears an object. Pass by reference...
    keys_to_delete = list(obj.keys())
    for k in keys_to_delete:
        if k == "/Length":
            continue
        del obj[k]
        '''
        try:
            del obj[k]
        except Exception:
            pass
        '''

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
    print("in_pdf: "+str(in_pdf))
    logger(f"[{now()}] Minimizing {in_pdf} -> {out_pdf} (max_pages={max_pages}, max_ops={max_ops}, image_thresh={image_threshold})")
    # try:

    with pikepdf.Pdf.open(in_pdf, allow_overwriting_input=True) as pdf: # Stuff...
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
        # print("pdf.objects: "+str(pdf.objects))
        for obj in pdf.objects: # list(pdf.objects.items()): # list(pdf.objects): # was originally list(pdf.objects.items()):
            print("obj: "+str(obj))
            try:
                # only consider stream objects with /Subtype /Image
                decode_flate_stream(obj, logger)
                if not isinstance(obj, pikepdf.Stream):
                    continue
                # many image XObjects have Subtype /Image
                subtype = obj.get("/Subtype")
                if subtype != pikepdf.Name("/Image"):
                    continue
                # stream size check
                # stream_len = len(obj.read_bytes() or b'')

                stream_len = obj["/Length"] # The stuff...


                if stream_len > image_threshold:
                    logger(f"   - Replacing large image object ({stream_len} bytes) with tiny PNG")

                    # Reset the dictionary (obj is a Stream, acts like a dict)
                    keys_to_delete = list(obj.keys())
                    for k in keys_to_delete:
                        try:
                            del obj[k]
                        except Exception:
                            pass

                    # Add new minimal dictionary entries
                    obj["/Type"] = pikepdf.Name("/XObject")
                    obj["/Subtype"] = pikepdf.Name("/Image")
                    obj["/Width"] = 1
                    obj["/Height"] = 1
                    obj["/ColorSpace"] = pikepdf.Name("/DeviceRGB")
                    obj["/BitsPerComponent"] = 8

                    # Overwrite stream data with tiny PNG bytes
                    # obj.write_bytes(TINY_PNG)
                    obj.write(TINY_PNG)


            except Exception as e:
                raise(e)
                # logger(f"   - Warning: while scanning image object {objnum}: {e}")
                logger(f"   - Warning: while scanning image object : {e}")

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
                                # c.clear()
                                print("c.keys() : "+str(c.keys()))
                                print(c)
                                print("type(c) : "+str(type(c)))
                                keys_to_delete = list(c.keys())
                                for k in keys_to_delete:
                                    print("k: "+str(k))
                                    if k != "/Length": # Do not try to delete length...
                                        del c[k]
                                print("poo1")
                                c["/Length"] == len(new_bytes)
                                print("poo2")
                                # c.write_bytes(new_bytes)
                                c.write(new_bytes)
                                new_contents.append(c)
                                truncated_streams += 1
                            else:
                                new_contents.append(c)
                        except Exception as e:
                            raise(e)
                            logger(f"     page {p_index}: error truncating stream array element: {e}")
                    page[ pikepdf.Name("/Contents") ] = new_contents
                elif isinstance(contents, pikepdf.Stream):
                    old_bytes = contents.read_bytes() or b''
                    new_bytes = truncate_content_stream_bytes(old_bytes, max_ops, logger)
                    # contents.clear()
                    clear_object(contents)
                    # contents.write_bytes(new_bytes)
                    # contents["/Length"] = len(new_bytes)
                    contents.write(new_bytes)
                    assert contents["/Length"] == len(new_bytes)
                    truncated_streams += 1
                else:
                    # weird type
                    continue
            except Exception as e:
                raise(e)
                logger(f"  - Warning: while truncating content on page {p_index}: {e}")

        # Save minimized PDF (let pikepdf rebuild xref and drop unreferenced objects)
        save_kwargs = {"linearize": True, "recompress_flate": False, "stream_decode_level": pikepdf.StreamDecodeLevel.none}
        # write to temp and then move to avoid partial writes
        tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
        
        pdf.save(str(tmp_out), **save_kwargs)
        tmp_out.replace(out_pdf)
        logger(f"Saved minimized PDF: {out_pdf} (replaced images: {replaced_images}, truncated streams: {truncated_streams})")
        return True, {"replaced_images": replaced_images, "truncated_streams": truncated_streams, "pages_kept": len(pdf.pages)}
    
    # except Exception as e:
    #     logger(f"Minimizer error for {in_pdf}: {e}")
    #     return False, {"error": str(e)}

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
        except Exception as e:
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
            logger(f"Minimization failed for {f.name} !!!")
            '''
            fallback_path = failed_dir / (f.name + ".bad")
            logger(f" - Minimization failed for {f.name}; copying original to {fallback_path}")
            try:
                safe_copy(dec_out, fallback_path, logger)
            except Exception as e:
                logger(f" - failed to copy fallback: {e}")
            '''

    logger("Done.")

if __name__ == "__main__":
    main()
