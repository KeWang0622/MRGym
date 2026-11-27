"""The public measurements the comparison uses: where each one is, under which licence, and how it is fetched.

Nothing here is downloaded unless ``run_all.py --fetch`` asks for it. A download goes to a cache folder OUTSIDE the
repository (``--cache``), is hashed, and is compared with the checksum recorded here; what the repository keeps is
the small extract under ``docs/physics/measured/`` (``STORED`` says what, and why that much). Licences were read at
the source on the date in ``retrieved``: the licence field of the Zenodo record, the licence the GitHub API reports
for the repository, the licence field of the Europe PMC record of an article.
"""
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile

AGENT = "python-urllib (research data fetch)"          # a generic client name; no request names this project
RETRIEVED = "2026-11-26"

SOURCES = dict(
    stupic=dict(
        source="Supporting information of Stupic et al. 2021: Tables S1 (NiCl2 array), S2 (MnCl2 array), S4 (water), file MRM-86-1194-s001.pdf",
        url="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8252537/supplementaryFiles",
        licence="CC BY 4.0 (the article; Europe PMC record PMC8252537)",
        citation="Stupic KF, Ainslie M, Boss MA, et al. A standard system phantom for magnetic resonance imaging. Magn Reson Med 2021;86:1194-1211. doi:10.1002/mrm.28779",
        sha256="149189eb04bf68064dc829bbb63db4b30f75f3fafc2a1f7557cf4002954106e7", sha256_of="MRM-86-1194-s001.pdf",
        stored="the values of the three tables, typed from the PDF (the file is a PDF: --fetch checks its checksum, it cannot re-read the tables)"),
    rrsg=dict(
        source="ISMRM 2020 reproducibility challenge, T1 maps of the NIST system phantom at 3 T: databases/3T_NIST_T1maps_database.csv and src/nist.py",
        url="https://raw.githubusercontent.com/rrsg2020/analysis/883abf38525f75f40f065060c4f555265d5af6a4/",
        files=("databases/3T_NIST_T1maps_database.csv", "src/nist.py"),
        licence="none (the repository has no licence file; the article that reports the data is CC BY)",
        citation="Boudreau M, Karakuzu A, Cohen-Adad J, et al. Repeat it without me: crowdsourcing the T1 mapping common ground via the ISMRM reproducibility challenge. "
                 "Magn Reson Med 2024;92:1115-1127. doi:10.1002/mrm.30111; data: github.com/rrsg2020/analysis",
        sha256="a07356c7479ed685eeb0c2e313886b04c2906d48526fbbaf22313c596f7263b9", sha256_of="databases/3T_NIST_T1maps_database.csv",
        stored="derived numbers only: per sphere the median and quartiles of the error and of the scatter over the submissions. The file is not stored"),
    hazen=dict(
        source="ACR large-phantom DICOM series of the hazen project: tests/data/acr, 14 series (T1 and T2 axial) of 9 scanners, 154 files",
        url="https://raw.githubusercontent.com/GSTT-CSC/hazen/dc7141d364070b69136ed9c5ae7d4f6e5a2f4836/tests/data/acr/",
        tree="https://api.github.com/repos/GSTT-CSC/hazen/git/trees/dc7141d364070b69136ed9c5ae7d4f6e5a2f4836?recursive=1",
        licence="Apache-2.0 (the licence of the repository; no separate statement for the images was found)",
        citation="hazen: quality assurance of MRI phantom images. Guy's and St Thomas' NHS Foundation Trust, Clinical Scientific Computing. github.com/GSTT-CSC/hazen",
        sha256="c84193020767a7679d997fa3ceaf43feeac7924e0f1e3710d4888b7310605262", sha256_of="the list `<series>/<file> <sha256>` of the 154 files, sorted, one per line",
        stored="derived numbers: the figures the QA functions of this package measure on slices 1, 5 and 7 of every series; and the pixels of ONE slice "
               "(a phantom 21 mm off centre), changed from the original: pixel array only, no header"),
    pdff=dict(
        source="Multi-site fat-water phantom MRI data: PDFF_Phantom_Data.zip, 28 acquisitions (6 sites, 3 vendors, 1.5 and 3 T, 2 protocols)",
        url="https://zenodo.org/api/records/48266/files/PDFF_Phantom_Data.zip/content",
        licence="CC0 1.0 (Zenodo record 48266)",
        citation="Hernando D, Sharma SD, Aliyari Ghasabeh M, et al. Multisite, multivendor validation of the accuracy and reproducibility of proton-density "
                 "fat-fraction quantification at 1.5T and 3T using a fat-water phantom. Magn Reson Med 2017;77:1516-1524. doi:10.1002/mrm.26228; "
                 "data: doi:10.5281/zenodo.48266",
        sha256="3dd517abfc8c1aefc3f5bac61833bc08dec663a3d20bb26a4588d10c59a9d4be", sha256_of="PDFF_Phantom_Data.zip",
        stored="the mean magnitude of the six echoes in each of the 11 vials of every acquisition that could be segmented, with the echo times"),
    gel=dict(
        source="Gel sphere at 3 T, product inversion-recovery TSE at 13 TI and single-echo spin echo at 10 TE: T1_T2_fitting_fBIRN_phantom.7z",
        url="https://zenodo.org/api/records/14778552/files/T1_T2_fitting_fBIRN_phantom.7z/content",
        licence="CC BY 4.0 (Zenodo record 14778552)",
        citation="Chen Q, Wehkamp N, Wan C, et al. Automated, open-source, vendor-independent quality assurance protocol based on the Pulseq framework. "
                 "Magn Reson Mater Phy 2025;38(3):533. doi:10.1007/s10334-025-01247-1; data: doi:10.5281/zenodo.14778552",
        sha256="dfec5389cac5d33e80d0edc9afeaffa37c8a68915c5a06f5d8fafe15119ea98f", sha256_of="T1_T2_fitting_fBIRN_phantom.7z",
        stored="the 23 mean signals of one region with their spread, and the header fields of the two series"),
    scholand=dict(
        source="NIST system phantom (model 130), T2 array, 3 T: single-echo spin echo at 12 TE, inversion-recovery spin echo at 11 TI, B1 map; raw k-space of 20 "
               "channels in BART format (data_GSM_t2, data_GSM_t1, data_05b_b1map)",
        url="https://zenodo.org/api/records/7654462/files/",
        files=("data_GSM_t2.cfl", "data_GSM_t2.hdr", "data_GSM_t1.cfl", "data_GSM_t1.hdr", "data_05b_b1map.cfl", "data_05b_b1map.hdr"),
        licence="CC BY 4.0 (Zenodo record 7654462)",
        citation="Scholand N, Wang X, Roeloffs V, Rosenzweig S, Uecker M. Quantitative MRI by nonlinear inversion of the Bloch equations. "
                 "Magn Reson Med 2023;90:520-538. doi:10.1002/mrm.29664; data: doi:10.5281/zenodo.7654462",
        sha256={"data_GSM_t2.cfl": "a6734ef08cfde51bb286e3bf6b82cc92af57e599c7e21ff25857d856efca64c5",
                "data_GSM_t1.cfl": "e3f42bcb54ea551313a00d12ea4e3248ca6b6d72e3bfe3e567f64e81f603a698",
                "data_05b_b1map.cfl": "f1367691a7a172bfc78e24edb1dc26d6b26df9f7df95e999a5022b7c791a53f5"},
        sha256_of="the three .cfl files",
        stored="per sphere the mean and spread of the signal at every TE and TI, the fitted place of the plate, and a second-order fit of the B1 map"),
    itis=dict(
        source="IT'IS Foundation tissue properties database, relaxation times (the two spreadsheets the page loads)",
        url="https://itis.swiss/_resources/vendor/oetiker/spreadbutler/fcgi/sb.fcgi/fetch?file=",
        files=("materialparameterdatabasecurrent20240604.xls", "materialparameterdatabasecurrent20220222c.xls"),
        licence="unclear: the pages carry a copyright notice and a citation request; no licence was found on them",
        citation="IT'IS Foundation. Tissue Properties Database, relaxation times. itis.swiss/virtual-population/tissue-properties/database/relaxation-times; doi:10.13099/VIP21000-05-0",
        sha256={"materialparameterdatabasecurrent20240604.xls": "807205b28413091f754c1dd50e7597d623f4a217ff0b4889e456096be893044a",
                "materialparameterdatabasecurrent20220222c.xls": "706bca58053b6257ddb9fb93a6f1e92ee08504e215b98cb3df24e2522461cf19"},
        sha256_of="the two spreadsheets as the page's table service returns them (JSON)",
        stored="nothing: the rung runs with --fetch only, and the repository keeps the counts it found"),
)


def provenance(name: str, **more) -> dict:
    """The fields every cached extract carries."""
    s = SOURCES[name]
    return dict(source=s["source"], url=s["url"], licence=s["licence"], citation=s["citation"], retrieved=RETRIEVED,
                sha256=more.pop("sha256", s["sha256"]), sha256_of=s["sha256_of"], stored=s["stored"], **more)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: str) -> str:
    """``url`` to ``path`` (kept when it is already there); returns the path."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(path + ".part", path)
    return path


def expect(path: str, sha256: str | None, what: str) -> str:
    """The checksum of ``path``; an error when it is not the recorded one."""
    got = sha256_file(path)
    if sha256 and got != sha256:
        raise RuntimeError(f"{what}: the checksum of the download ({got}) is not the recorded one ({sha256}); the source has changed")
    return got


def fetch(name: str, cache: str) -> str:
    """Fetch the originals of one source into ``cache/<name>/`` and check them; returns that folder."""
    s, d = SOURCES[name], os.path.join(cache, name)
    os.makedirs(d, exist_ok=True)
    if name == "stupic":
        z = download(s["url"], os.path.join(d, "supplement.zip"))
        with zipfile.ZipFile(z) as f:
            member = next(m for m in f.namelist() if m.endswith(s["sha256_of"]))
            with f.open(member) as src, open(os.path.join(d, s["sha256_of"]), "wb") as dst:      # one named member, written under a fixed name
                shutil.copyfileobj(src, dst)
        expect(os.path.join(d, s["sha256_of"]), s["sha256"], name)
    elif name in ("rrsg", "scholand"):
        for rel in s["files"]:
            p = download(s["url"] + rel + ("/content" if name == "scholand" else ""), os.path.join(d, os.path.basename(rel)))
            want = s["sha256"].get(rel) if isinstance(s["sha256"], dict) else (s["sha256"] if rel == s["sha256_of"] else None)
            if want:
                expect(p, want, f"{name}: {rel}")
    elif name == "itis":
        for rel in s["files"]:
            expect(download(s["url"] + rel, os.path.join(d, rel + ".json")), s["sha256"][rel], f"{name}: {rel}")
    elif name == "hazen":
        with open(download(s["tree"], os.path.join(d, "tree.json")), encoding="utf-8") as f:
            tree = json.load(f)["tree"]
        for t in tree:
            if t["type"] == "blob" and t["path"].startswith("tests/data/acr/") and t["path"].count("/") == 4:
                series, fname = t["path"].split("/")[3:]
                if series in ACR_SERIES and "/" not in fname and ".." not in fname:
                    download(s["url"] + f"{series}/{fname}", os.path.join(d, series, fname))
        got = manifest_sha256(d, [f"{series}/{f}" for series in ACR_SERIES for f in sorted(os.listdir(os.path.join(d, series)))])
        if got != s["sha256"]:
            raise RuntimeError(f"hazen: the checksum of the downloaded series ({got}) is not the recorded one ({s['sha256']}); the source has changed")
    elif name == "pdff":
        z = download(s["url"], os.path.join(d, "PDFF_Phantom_Data.zip"))
        expect(z, s["sha256"], name)
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                base = os.path.basename(m)
                if m.endswith(".mat") and "/datasets/" in m and base and not base.startswith("."):
                    with f.open(m) as src, open(os.path.join(d, base), "wb") as dst:           # flat, under its own base name
                        shutil.copyfileobj(src, dst)
    elif name == "gel":
        z = download(s["url"], os.path.join(d, "T1_T2_fitting_fBIRN_phantom.7z"))
        expect(z, s["sha256"], name)
        if not os.path.isdir(os.path.join(d, "T1_T2_fitting_fBIRN_phantom")):
            unpack_7z(z, d)
    return d


def unpack_7z(archive: str, into: str) -> None:
    """Unpack a .7z archive. Python has no reader for the container: ``py7zr`` if it is installed, else a ``7z`` /
    ``7za`` / ``7zr`` / ``bsdtar`` program; otherwise an error that says so (the stored extract is then used)."""
    try:
        import py7zr
        with py7zr.SevenZipFile(archive) as f:
            f.extractall(into)
        return
    except ImportError:
        pass
    for prog, args in (("7z", ["x", "-y", f"-o{into}", archive]), ("7za", ["x", "-y", f"-o{into}", archive]), ("7zr", ["x", "-y", f"-o{into}", archive]),
                       ("bsdtar", ["-xf", archive, "-C", into])):
        exe = shutil.which(prog)
        if exe:
            subprocess.run([exe, *args], check=True, stdout=subprocess.DEVNULL)
            return
    raise RuntimeError("the gel data set is a .7z archive: install py7zr (pip) or a 7z program, or unpack it by hand into "
                       f"{into}/T1_T2_fitting_fBIRN_phantom")


ACR_SERIES = ("GE_Artist_1.5T_T1", "GE_Artist_1.5T_T2", "GE_MR450W_1.5T_T1", "GE_MR450W_1.5T_T2", "GE_Signa_1.5T_T1", "GE_Signa_3T_T1", "GE_Signa_3T_T2",
              "Philips3TdStream", "PhilipsAchieva", "SiemensSolaFit", "Siemens_Aera_1.5T_T1", "Siemens_Aera_1.5T_T2", "Siemens_MagnetomSkyra_3T_T1",
              "Siemens_MagnetomSkyra_3T_T2")


def manifest_sha256(folder: str, names) -> str:
    """Checksum of a set of files: sha256 of the lines ``<name> <sha256 of the file>``, sorted."""
    lines = sorted(f"{n} {sha256_file(os.path.join(folder, n))}" for n in names)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
