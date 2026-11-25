# References

Every paper named in `src/mrgym`, with its full record, the file that uses it and how far it was checked.

**Checked on 2026-11-24.** Each DOI below was fetched from Crossref on that day
(`https://api.crossref.org/works/<DOI>`). Authors, title, journal, year, volume, issue and pages are those of the
Crossref record. All 23 records resolved. One record is incomplete and says so on its line.

**No paper body was read for this release.** The check covers bibliographic records only. The column "Checked
against" says what the value or formula in the package was compared with:

- **abstract**: the value is stated in the abstract of the paper (tag [A] in `src/mrgym/scanner/tissue.py`);
- **table, not re-checked**: the value is attributed to a table of the paper and was not compared with the full
  text (tag [T]);
- **not compared with the paper**: the formula is the one the paper is known for, written down by the authors of
  this package. Where a row of `mrgym validate` checks it numerically against a separate calculation, the row is
  named. That is a check of the code, not of the citation.

To repeat the check for one entry: `curl -s https://api.crossref.org/works/10.1002/mrm.21120`.

## Signal models and steady states

| Reference | Record | Used in | Checked against |
|---|---|---|---|
| Ernst & Anderson 1966 | R. R. Ernst, W. A. Anderson. Application of Fourier transform spectroscopy to magnetic resonance. *Review of Scientific Instruments* 37(1):93-102, 1966. doi:10.1063/1.1719961 | `scanner/signal.py` (`spgr`), `validate.py` (S1, S4) | not compared with the paper; S1, S4 |
| Zur 1991 | Y. Zur, M. L. Wood, L. J. Neuringer. Spoiling of transverse magnetization in steady-state sequences. *Magnetic Resonance in Medicine* 21(2):251-263, 1991. doi:10.1002/mrm.1910210210 | `validate.py` (S9) | not compared with the paper; S9 shows only that the served signal differs from the Ernst formula |
| Preibisch & Deichmann 2009 | C. Preibisch, R. Deichmann. Influence of RF spoiling on the stability and accuracy of T1 mapping based on spoiled FLASH with varying flip angles. *Magnetic Resonance in Medicine* 61(1):125-135, 2009. doi:10.1002/mrm.21776 | `validate.py` (S9) | not compared with the paper |
| Yarnykh 2007 | V. L. Yarnykh. Actual flip-angle imaging in the pulsed steady state: a method for rapid three-dimensional mapping of the transmitted radiofrequency field. *Magnetic Resonance in Medicine* 57(1):192-200, 2007. doi:10.1002/mrm.21120 | `scanner/signal.py` (`afi`), `validate.py` (S2) | not compared with the paper; S2 |
| Nehrke 2009 | K. Nehrke. On the steady-state properties of actual flip angle imaging (AFI). *Magnetic Resonance in Medicine* 61(1):84-92, 2009. doi:10.1002/mrm.21592 | `scanner/epg.py` (spoiler moments and RF phase increments of `afi` in the ratio TR1 : TR2) | not compared with the paper; S5, S6 |
| Weigel 2015 | M. Weigel. Extended phase graphs: dephasing, RF pulses, and echoes - pure and simple. *Journal of Magnetic Resonance Imaging* 41(2):266-295, 2015. doi:10.1002/jmri.24619 | `scanner/epg.py` (state notation), `scanner/signal.py` | not compared with the paper; S5 to S8 |
| Weigel 2010 | M. Weigel, S. Schwenk, V. G. Kiselev, K. Scheffler, J. Hennig. Extended phase graphs with anisotropic diffusion. *Journal of Magnetic Resonance* 205(2):276-285, 2010. doi:10.1016/j.jmr.2010.05.011 | `scanner/bloch.py` (`rf_spoiled_epg_stepping`); the same b-factors in `scanner/epg.py` (`_interval`) | not compared with the paper; S6 checks the solver with the same formula, not the formula |
| Hänicke & Vogel 2003 | W. Hänicke, H. U. Vogel. An analytical solution for the SSFP signal in MRI. *Magnetic Resonance in Medicine* 49(4):771-775, 2003. doi:10.1002/mrm.10410 | `validate.py` (`ssfp_fid`, S7) | not compared with the paper; S7 |
| Hennig 1988 | J. Hennig. Multiecho imaging sequences with low refocusing flip angles. *Journal of Magnetic Resonance* 78(3):397-407, 1988. doi:10.1016/0022-2364(88)90128-X | `scanner/signal.py` (`se`: refocusing efficiency sin²(β / 2)), `validate.py` (S12) | not compared with the paper; S12 |
| Bernstein, King & Zhou 2004 | M. A. Bernstein, K. F. King, X. J. Zhou. *Handbook of MRI Pulse Sequences*. Elsevier, 2004. ISBN 978-0-12-092861-3. doi:10.1016/B978-0-12-092861-3.X5000-6. The Crossref record gives title, publisher, year and ISBN and lists no authors; the names are those cited in the source. | `scanner/signal.py` (`se`, "ch. 14") | not compared with the book; the chapter number is not verified |
| Freeman & Hill 1971 | R. Freeman, H. D. W. Hill. Phase and intensity anomalies in Fourier transform NMR. *Journal of Magnetic Resonance* 4(3):366-383, 1971. doi:10.1016/0022-2364(71)90047-3 | `scanner/signal.py` (`bssfp`), `validate.py` (S14) | not compared with the paper; S14 |
| Scheffler & Lehnhardt 2003 | K. Scheffler, S. Lehnhardt. Principles and applications of balanced SSFP techniques. *European Radiology* 13(11):2409-2418, 2003. doi:10.1007/s00330-003-1957-x | `scanner/signal.py` (`bssfp`), `validate.py` (S14) | not compared with the paper; S14 |

## Image formation, fields and noise

| Reference | Record | Used in | Checked against |
|---|---|---|---|
| Jezzard & Balaban 1995 | P. Jezzard, R. S. Balaban. Correction for geometric distortion in echo planar images from B0 field variations. *Magnetic Resonance in Medicine* 34(1):65-73, 1995. doi:10.1002/mrm.1910340111 | `validate.py` (K12) | not compared with the paper; K12 |
| Gudbjartsson & Patz 1995 | H. Gudbjartsson, S. Patz. The Rician distribution of noisy MRI data. *Magnetic Resonance in Medicine* 34(6):910-914, 1995. doi:10.1002/mrm.1910340618 | `validate.py` (K16) | not compared with the paper; K16 |
| Schenck 1996 | J. F. Schenck. The role of magnetic susceptibility in magnetic resonance imaging: MRI magnetic compatibility of the first and second kinds. *Medical Physics* 23(6):815-850, 1996. doi:10.1118/1.597854 | `scanner/world.py` (field of an air sphere, susceptibility difference 9.4 ppm) | not compared with the paper; no row of `mrgym validate` covers the dipole field |
| Beatty 2005 | P. J. Beatty, D. G. Nishimura, J. M. Pauly. Rapid gridding reconstruction with a minimal oversampling ratio. *IEEE Transactions on Medical Imaging* 24(6):799-808, 2005. doi:10.1109/TMI.2005.848376 | `scanner/kspace.py` (Kaiser-Bessel kernel, width 4, shape 8.996) | not compared with the paper; the displacement operator that uses the kernel is covered by K2 and K5 |

## Tissue values (`src/mrgym/scanner/tissue.py`)

| Reference | Record | Used for | Checked against |
|---|---|---|---|
| Wright 2008 | P. J. Wright, O. E. Mougin, J. J. Totman, A. M. Peters, M. J. Brookes, R. Coxon, P. E. Morris, M. Clemence, S. T. Francis, R. W. Bowtell, P. A. Gowland. Water proton T1 measurements in brain tissue at 7, 3, and 1.5 T using IR-EPI, IR-TSE, and MPRAGE: results and optimization. *Magnetic Resonance Materials in Physics, Biology and Medicine* 21(1-2):121-130, 2008. doi:10.1007/s10334-008-0104-8 | T1 of white matter, grey matter, putamen and caudate at 1.5 and 3 T | abstract |
| Rooney 2007 | W. D. Rooney, G. Johnson, X. Li, E. R. Cohen, S.-G. Kim, K. Ugurbil, C. S. Springer. Magnetic field and tissue dependencies of human brain longitudinal 1H2O relaxation in vivo. *Magnetic Resonance in Medicine* 57(2):308-318, 2007. doi:10.1002/mrm.21122 | T1 of cerebrospinal fluid | abstract |
| Wansapura 1999 | J. P. Wansapura, S. K. Holland, R. S. Dunn, W. S. Ball. NMR relaxation times in the human brain at 3.0 tesla. *Journal of Magnetic Resonance Imaging* 9(4):531-538, 1999. doi:`10.1002/(SICI)1522-2586(199904)9:4<531::AID-JMRI4>3.0.CO;2-L` | named for the T2 of grey and white matter at 3 T | not used as a source since engine 0.2.0: a review of this release reported that the abstract prints grey 80 / white 110 ms, the opposite assignment to the table's 95 / 75 ms; the two values are tagged [C]. The abstract was not re-read for this change |
| Peters 2007 | A. M. Peters, M. J. Brookes, F. G. Hoogenraad, P. A. Gowland, S. T. Francis, P. G. Morris, R. Bowtell. T2* measurements in human brain at 1.5, 3 and 7 T. *Magnetic Resonance Imaging* 25(6):748-753, 2007. doi:10.1016/j.mri.2007.02.014 | T2* of grey and white matter | table, not re-checked |
| Stanisz 2005 | G. J. Stanisz, E. E. Odrobina, J. Pun, M. Escaravage, S. J. Graham, M. J. Bronskill, R. M. Henkelman. T1, T2 relaxation and magnetization transfer in tissue at 3T. *Magnetic Resonance in Medicine* 54(3):507-512, 2005. doi:10.1002/mrm.20605 | T1 and T2 of muscle, blood and spinal cord | table, not re-checked |
| de Bazelaire 2004 | C. M. J. de Bazelaire, G. D. Duhamel, N. M. Rofsky, D. C. Alsop. MR imaging relaxation times of abdominal and pelvic tissues measured in vivo at 3.0 T: preliminary results. *Radiology* 230(3):652-659, 2004. doi:10.1148/radiol.2303021331 | T1 and T2 of abdominal organs, fat and marrow | 3 T values: abstract; 1.5 T values: table, not re-checked |
| Hamilton 2011 | G. Hamilton, T. Yokoo, M. Bydder, I. Cruite, M. E. Schroeder, C. B. Sirlin, M. S. Middleton. In vivo characterization of the liver fat 1H MR spectrum. *NMR in Biomedicine* 24(7):784-790, 2011. doi:10.1002/nbm.1622 | six-peak fat spectrum: positions and amplitudes | table, not re-checked |

The source lists Wansapura 1999 without a DOI; the DOI above is the one Crossref resolves to that record. Every
DOI and every volume and page written in the source code agrees with its Crossref record.

Values tagged [C] in the tissue table have no source: T2 and T2* of deep grey matter, skull, skin, bowel fluid,
T2* of abdominal tissue and all lesion classes. Some numbers carry no tag: the diffusion coefficients and most
proton densities. Treat them as [C]. [limits.md](limits.md), block 14, says what that means.

## Added with engine 0.2.0: records not fetched

The papers below are named in the source since engine 0.2.0 (slice selection, motion, the perturbation table, the
external ladder). **Their bibliographic records were not fetched from Crossref and no body was read for this
release.** What the package takes from each is a statement or a number quoted from its abstract in the design notes
of the release; treat every row as "to be verified" until the check at the top of this page is repeated for it.

| Reference | What is known of the record | Used in | What is taken from it |
|---|---|---|---|
| Pauly 1991 | J. Pauly, P. Le Roux, D. Nishimura, A. Macovski. Parameter relations for the Shinnar-Le Roux selective excitation pulse design algorithm. *IEEE Transactions on Medical Imaging*, 1991. doi:10.1109/42.75611 (not fetched) | `scanner/rf.py` (hard-pulse cascade as Cayley-Klein parameters) | the forward transform; written down by the authors of this package and checked numerically by R1, R2, L1.2 |
| Malik 2011 | S. J. Malik and others, 2011 (journal, volume and DOI not recorded) | `scanner/rf.py`, `perturb.py` | the statement that actual flip-angle imaging with 2D encoding has systematic errors from the slice profile |
| Dieringer 2014 | M. A. Dieringer and others, 2014 (journal, volume and DOI not recorded) | `validate.py` (R12), `tools/external/run_all.py` (L3.5), example 13 | the bias of uncorrected 2D variable-flip-angle T1, (-63 +- 1.5) % |
| Svedin & Parker 2017 | B. T. Svedin, D. L. Parker, 2017 (journal, volume and DOI not recorded) | the same places | 40 to 80 % underestimation of T1 without profile correction |
| Zaitsev 2015 | M. Zaitsev, J. Maclaren, M. Herbst. Motion artifacts in MRI: a complex problem with many partial solutions. *Journal of Magnetic Resonance Imaging*, 2015 (PMC4517972; DOI not recorded) | `validate.py` (K25), `perturb.py` | periodic modulation of k-space lines gives coherent ghosts; the spacing lines x f x TR x averages is derived in `validate.py` and checked there |
| Newcombe 1998 | R. G. Newcombe, *Statistics in Medicine*, 1998 (volume, pages and DOI not recorded) | `perturb.py` (`paired_difference`) | the interval for a difference of paired proportions built from Wilson intervals; the formula is written down by the authors of this package and its coverage is checked by a simulation in `tests/test_validity.py` |
| Yarnykh 2010 | V. L. Yarnykh, *Magnetic Resonance in Medicine* 63:1610, 2010 (DOI not recorded) | `tools/external/run_all.py` (L3.1 to L3.4) | the operating points and the statement "below 2 %" of its abstract |
| Stupic 2021 | K. F. Stupic and others, 2021 (PMC8252537; journal and DOI not recorded) | `tools/external/run_all.py` (rung L4, not built) | named as the standard object a later rung needs |

## Reference objects and the examined outline

Documents that were opened and read for the phantom worlds (`anatomy.acr`, `anatomy.nist_t1`), the QA command
(`mrgym/qa.py`) and the competency map (`mrgym/coverage.py`), on 2026-11-24 and 2026-11-25. Unlike the rows above,
these were read in the body: each number taken from them carries a tag and a section in the source file, and what
they do not give is tagged as declared. What was drawn from them and what was not: [phantoms.md](phantoms.md).

| Reference | Record | Used in | Checked against |
|---|---|---|---|
| ACR phantom guidance | American College of Radiology. *Large and Medium Phantom Test Guidance for the ACR MRI Accreditation Program*. The version dated 10.19.2022 (42 pages), linked from the accreditation support article "Large/Medium Phantom Testing: MRI (Revised 2-6-26)", accreditationsupport.acr.org, article 11000061035 | `scanner/anatomy.py` (`ACR`: dimensions, hole sizes, contrasts), `qa.py` (`SERIES`, `LIMITS`, the definitions of the measurements, the causes of failure named per fault) | **the document, read in full**; tag [G] with the section or table. The relaxation times of the solution, the layout inside a section and the bandwidth are not in it (tag [D]) |
| Stupic 2021 | K. F. Stupic and others. A standard system phantom for magnetic resonance imaging. *Magnetic Resonance in Medicine* 86(3):1194-1211, 2021. doi:10.1002/mrm.28779 (PMC8252537) | `scanner/anatomy.py` (`NIST_T1`: shell and sphere diameters, the arrangement of the 14 spheres, T1 and T2 of the Ni-12 solution at 3 T) | **the article body** (PMC full text). The supporting Tables S1 to S3 (positions, 1.5 T values, T2 of every sphere) were not opened |
| ISMRM reproducibility challenge, analysis code | `rrsg2020/analysis` on GitHub, file `src/nist.py`: reference T1 of the NiCl2 array at 20 degrees C for phantoms with serial number below 0043 (the file names its source as a document of the phantom's maker) | `scanner/anatomy.py` (`NIST_T1["t1_ms"]`) | **the file**. Its twelfth value (44.53 ms) is the Ni-12 value of Stupic 2021 at 3 T; the maker's document itself was not opened |
| ARRT content specification | American Registry of Radiologic Technologists. *Examination Content Specifications: Magnetic Resonance Imaging*. Board approved January 2024, implementation date February 1, 2025. Image Production: 1.C, 1.D, 2.A, 2.C | `coverage.py` (the rows of the competency map) | **the document**: the names and the numbering of the items. Used as a map of topics; no examination item is reproduced |

## Not cited in the source

Statements that the package makes without a reference, so that a reader does not look for one:

- the Gibbs overshoot of 8.95 % (row K8), the Rayleigh moments 1.2533 and 0.6551 (row K16) and the null point
  T1 ln 2 (row S13) are standard results quoted as numbers;
- the proton gyromagnetic ratio 42.577478518 MHz/T carries no citation in the code;
- the sizes of the system errors and of the noise are choices of the package ([limits.md](limits.md), block 15).
