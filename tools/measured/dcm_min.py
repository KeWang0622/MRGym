"""A minimal DICOM reader (uncompressed little-endian, explicit or implicit VR): the header fields a protocol
comparison needs and the pixel array. The package has no DICOM dependency; this reads data only and executes
nothing from the file. Used by ``extract.py`` on the public series of the comparison with measurements."""
import struct

import numpy as np

LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"OD", b"OL", b"UC", b"UR", b"OV", b"SV", b"UV"}
WANT = {
    (0x0008, 0x0070): "Manufacturer", (0x0008, 0x1090): "Model", (0x0008, 0x103E): "SeriesDescription", (0x0008, 0x0008): "ImageType",
    (0x0018, 0x0020): "ScanningSequence", (0x0018, 0x0021): "SequenceVariant", (0x0018, 0x0023): "MRAcquisitionType", (0x0018, 0x0024): "SequenceName",
    (0x0018, 0x0050): "SliceThickness", (0x0018, 0x0080): "TR", (0x0018, 0x0081): "TE", (0x0018, 0x0083): "Averages",
    (0x0018, 0x0087): "FieldStrength", (0x0018, 0x0088): "SliceSpacing", (0x0018, 0x0091): "EchoTrainLength", (0x0018, 0x0095): "PixelBandwidth",
    (0x0018, 0x0093): "PercentSampling", (0x0018, 0x0094): "PercentPhaseFOV", (0x0018, 0x1250): "ReceiveCoil", (0x0018, 0x1310): "AcquisitionMatrix",
    (0x0018, 0x1312): "PhaseDirection", (0x0018, 0x1314): "FlipAngle", (0x0018, 0x0084): "ImagingFrequency",
    (0x0020, 0x0013): "InstanceNumber", (0x0020, 0x0032): "Position", (0x0020, 0x0037): "Orientation", (0x0020, 0x1041): "SliceLocation",
    (0x0028, 0x0010): "Rows", (0x0028, 0x0011): "Columns", (0x0028, 0x0030): "PixelSpacing", (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0103): "PixelRepresentation", (0x0028, 0x1052): "RescaleIntercept", (0x0028, 0x1053): "RescaleSlope", (0x0028, 0x0008): "Frames",
    (0x0002, 0x0010): "TransferSyntax", (0x7FE0, 0x0010): "PixelData",
    (0x0018, 0x9082): "EffectiveTE", (0x0018, 0x9079): "InversionTimes", (0x0018, 0x9058): "PixelBandwidth9", (0x0018, 0x9241): "EchoTrainLength9",
}
IMPLICIT_INT = {(0x0028, 0x0010), (0x0028, 0x0011), (0x0028, 0x0100), (0x0028, 0x0103), (0x0018, 0x1310)}


def _skip_sequence(b, pos, explicit):
    """Skip a sequence of undefined length that starts at ``pos`` (after its header); return the position after it."""
    depth = 1
    while depth and pos < len(b):
        g, e = struct.unpack_from("<HH", b, pos)
        if g == 0xFFFE:
            (ln,) = struct.unpack_from("<I", b, pos + 4)
            pos += 8
            if e == 0xE0DD:
                depth -= 1
            elif e == 0xE000 and ln != 0xFFFFFFFF:
                pos += ln
            continue
        pos, vr, ln, _ = _header(b, pos, explicit)
        if ln == 0xFFFFFFFF:
            depth += 1
        else:
            pos += ln
    return pos


def _header(b, pos, explicit):
    g, e = struct.unpack_from("<HH", b, pos)
    if explicit and g != 0xFFFE:
        vr = b[pos + 4:pos + 6]
        if vr in LONG_VR:
            (ln,) = struct.unpack_from("<I", b, pos + 8)
            return pos + 12, vr, ln, (g, e)
        (ln,) = struct.unpack_from("<H", b, pos + 6)
        return pos + 8, vr, ln, (g, e)
    (ln,) = struct.unpack_from("<I", b, pos + 4)
    return pos + 8, None, ln, (g, e)


def read(path, flat=False):
    """``flat``: walk into sequences instead of skipping them (enhanced multi-frame objects keep the timing and the
    pixel spacing inside functional-group sequences); the first occurrence of a field is kept. Tried automatically
    when the pixel spacing is not at the top level."""
    out = _read(path, flat)
    if not flat and "PixelSpacing" not in out:
        out = _read(path, True)
    return out


def _read(path, flat):
    with open(path, "rb") as f:
        b = f.read()
    if b[128:132] != b"DICM":
        raise ValueError("no DICM preamble")
    pos, out, explicit = 132, {}, True
    meta_end = None
    while pos < len(b) - 8:
        g = struct.unpack_from("<H", b, pos)[0]
        if g != 0x0002 and meta_end is None:
            meta_end = pos
            ts = out.get("TransferSyntax", "1.2.840.10008.1.2.1")
            if ts == "1.2.840.10008.1.2":
                explicit = False
            elif ts != "1.2.840.10008.1.2.1":
                raise ValueError(f"transfer syntax {ts} is not uncompressed little-endian")
        pos, vr, ln, tag = _header(b, pos, explicit or meta_end is None)
        if flat and tag[0] == 0xFFFE:
            continue                                   # item or delimiter: its content is parsed as elements
        if flat and (vr == b"SQ" or (ln == 0xFFFFFFFF and vr in (None, b"SQ"))):
            continue
        if ln == 0xFFFFFFFF:
            pos = _skip_sequence(b, pos, explicit)
            continue
        val = b[pos:pos + ln]
        pos += ln
        name = WANT.get(tag)
        if not name or (flat and name in out):
            continue
        if name == "PixelData":
            out[name] = val
        elif vr == b"FD":
            v = struct.unpack("<" + "d" * (ln // 8), val)
            out[name] = v[0] if len(v) == 1 else list(v)
        elif vr == b"US" or (vr is None and tag in IMPLICIT_INT):
            v = struct.unpack("<" + "H" * (ln // 2), val)
            out[name] = v[0] if len(v) == 1 else list(v)
        else:
            out[name] = val.decode("latin-1").strip(" \x00")
    dt = {(16, 0): "<u2", (16, 1): "<i2", (8, 0): "u1"}[(out["BitsAllocated"], out.get("PixelRepresentation", 0))]
    px = np.frombuffer(out.pop("PixelData"), dt)
    fr = int(out.get("Frames", 1))
    px = px[:fr * out["Rows"] * out["Columns"]].reshape(fr, out["Rows"], out["Columns"]).astype(float)
    out["pixels"] = px * float(out.get("RescaleSlope", 1) or 1) + float(out.get("RescaleIntercept", 0) or 0)
    return out

