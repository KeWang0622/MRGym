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

## Receive arrays and parallel imaging: records from Europe PMC, bodies read where open

Added with `scanner/array.py` and `scanner/recon.py` on 2026-11-26. These records are those of the Europe PMC
search interface (title, authors, journal, year, volume, issue, pages, DOI), not of Crossref, and are not among
the 23 counted above. Unlike the rows above, the body of most of these papers was fetched and the passage that is
used was read; the column says which. "Not compared with the paper" has the meaning given at the top of this page:
the formula is supported only by the rows named ([array.md](array.md), section 7). The errata of Kellman 2005 (58(1):211-2,
2007) and of Constantinides 1997 (52(1):219, 2004) exist and were not obtained.

| Reference | Record | Used in | Checked against |
|---|---|---|---|
| Roemer 1990 | P. B. Roemer, W. A. Edelstein, C. E. Hayes, S. P. Souza, O. M. Mueller. The NMR phased array. *Magnetic Resonance in Medicine* 16(2):192-225, 1990. doi:10.1002/mrm.1910160203 | `scanner/recon.py` (`roemer`, `roemer_noise`, `sos`), `scanner/array.py` (overlap of neighbouring loops, the noise covariance) | **the abstract only** (overlapping neighbours, an algorithm for optimum SNR). The matched filter, the bound on the root sum of squares and the loss integral were **not compared with the paper**; rows C1, C3, C7, C9, C10 |
| Pruessmann 1999 | K. P. Pruessmann, M. Weiger, M. B. Scheidegger, P. Boesiger. SENSE: sensitivity encoding for fast MRI. *Magnetic Resonance in Medicine* 42(5):952-962, 1999. doi:10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S | `scanner/recon.py` (`sense_plan`, `sense_unfold`, `estimate_maps`) | **the article body** (an archived copy): the unfolding matrix (equation 2), the g-factor (23), the SNR law (24), maps by division by a sum-of-squares or a body-coil image with a local polynomial fit (here: a weighted Gaussian smoothing instead); rows P1 to P24 |
| Pruessmann 2001 | K. P. Pruessmann, M. Weiger, P. Boernert, P. Boesiger. Advances in sensitivity encoding with arbitrary k-space trajectories. *Magnetic Resonance in Medicine* 46:638-651, 2001 (record from the reference list of Kellman 2005; DOI not recorded) | `scanner/recon.py` (`sense_cg`) | not fetched; **not compared with the paper**; row P2 |
| Griswold 2002 | M. A. Griswold, P. M. Jakob, R. M. Heidemann, M. Nittka, V. Jellus, J. Wang, B. Kiefer, A. Haase. Generalized autocalibrating partially parallel acquisitions (GRAPPA). *Magnetic Resonance in Medicine* 47(6):1202-1210, 2002. doi:10.1002/mrm.10171 | `scanner/recon.py` (`grappa`: a seam, not built; `calibration_block`) | **the article body** (an archived copy): weights fitted on a block of calibration lines in the centre of k-space |
| Walsh 2000 | D. O. Walsh, A. F. Gmitro, M. W. Marcellin. Adaptive reconstruction of phased array MR imagery. *Magnetic Resonance in Medicine* 43(5):682-690, 2000. doi:10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G | `scanner/recon.py` (`adaptive`) | **the submitted manuscript** (an archived copy, not the printed article): the filter is the eigenvector of the largest eigenvalue of the product of the inverse noise correlation and the signal correlation, estimated over local regions; rows C11, C12 |
| Kellman & McVeigh 2005 | P. Kellman, E. R. McVeigh. Image reconstruction in SNR units: a general method for SNR measurement. *Magnetic Resonance in Medicine* 54(6):1439-1447, 2005. doi:10.1002/mrm.20713 (PMC2570032) | `scanner/recon.py` (`roemer(snr_units=True)`, `sense_snr_units`, `prewhiten`, `chi_moments`) | **the article body** (PMC): whitening to virtual channels of unit variance, the scale 1 / sqrt(N) of a transform over N samples, the SNR-scaled unfolding, the bias of the background for 1 to 32 channels (compared with `chi_moments` in `tools/audit/array_rederive.py`). The noise-equivalent bandwidth of a receiver filter is not modelled: the engine's noise is white |
| Robson 2008 | P. M. Robson, A. K. Grant, A. J. Madhuranthakam, R. Lattanzi, D. K. Sodickson, C. A. McKenzie. Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions. *Magnetic Resonance in Medicine* 60(4):895-907, 2008. doi:10.1002/mrm.21728 (PMC2838249) | `scanner/recon.py` (`pseudo_replica`, `noise_covariance`) | **the article body** (PMC): replicas of correctly scaled and correlated noise through the reconstruction, psi with the factor 1 / (2 N) (equation 10); rows N1, C7, P6, P11, P16, P21 |
| Constantinides 1997 | C. D. Constantinides, E. Atalar, E. R. McVeigh. Signal-to-noise measurements in magnitude images from NMR phased arrays. *Magnetic Resonance in Medicine* 38(5):852-857, 1997. doi:10.1002/mrm.1910380524 (PMC2570034) | `scanner/recon.py` (`chi_moments`) | **the article body** (PMC): the central chi density with 2n degrees of freedom (equation 4), second moment 2 n sigma^2, the printed means and standard deviations for 2 and 4 channels (compared in `tools/audit/array_rederive.py`); rows N4, N5, N7 |
| Breuer 2005 | F. A. Breuer, M. Blaimer, R. M. Heidemann, M. F. Mueller, M. A. Griswold, P. M. Jakob. Controlled aliasing in parallel imaging results in higher acceleration (CAIPIRINHA) for multi-slice imaging. *Magnetic Resonance in Medicine* 53(3):684-691, 2005. doi:10.1002/mrm.20401 | named in `docs/physics/array.md` as not built | **the article body** (an archived copy) |
| Breuer 2009 | F. A. Breuer, S. A. Kannengiesser, M. Blaimer, N. Seiberlich, P. M. Jakob, M. A. Griswold. General formulation for quantitative G-factor calculation in GRAPPA reconstructions. *Magnetic Resonance in Medicine* 62(3):739-746, 2009. doi:10.1002/mrm.22066 | named in `docs/physics/array.md` as not built | **the abstract only** |

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

## Measured data and the papers of the comparison with measurements

Used by `tools/measured/` ([measured.md](measured.md)), added on 2026-11-26 and 2026-11-27. The licence of each data
set was read at its source (the licence field of the Zenodo record, the licence the GitHub API reports, the licence
field of the Europe PMC record); the full citations and the checksums of the originals are in
`tools/measured/sources.py` and in the files under `docs/physics/measured/`. "Read" says how far each was read for
this work.

| Reference | Record | Used in | Read |
|---|---|---|---|
| Stupic 2021, supporting information | Tables S1, S2, S4, S10 of the supporting information of the article above (file `MRM-86-1194-s001.pdf`); the article is CC BY | `scanner/anatomy.py` (`NIST_S1`, `NIST_S2`, `NIST_XZ_MM`, `NIST_WATER_RATE_PER_S`, `nist_t2`), rungs M0, M2, M9 | **the tables**, compared digit by digit with the typed values; and the sentence of the article on the temperature coefficients of the MnCl2 array |
| ISMRM 2020 T1 challenge | M. Boudreau and others, *Magnetic Resonance in Medicine* 92:1115, 2024, doi:10.1002/mrm.30111 (CC BY); data: `rrsg2020/analysis` on GitHub, `databases/3T_NIST_T1maps_database.csv` and `src/nist.py` (no licence) | rung M1 | **the two files**; the article was not read in the body |
| Keenan 2021 | K. E. Keenan and others, *PLOS ONE* 16:e0252966, 2021, doi:10.1371/journal.pone.0252966 (CC0) | rung M2 | the two sentences quoted in rows M2.1 and M2.2, found in the text of the article |
| Dieringer 2014 | M. A. Dieringer and others, *PLOS ONE* 9:e91318, 2014, doi:10.1371/journal.pone.0091318 (CC BY) | rung M2b (and `validate.py` R12, `tools/external/run_all.py` L3.5) | the numbers of its phantom experiment and the description of its pulse, found in the text |
| Ben-Eliezer 2015 | N. Ben-Eliezer, D. K. Sodickson, K. T. Block, *Magnetic Resonance in Medicine* 73:809, 2015, doi:10.1002/mrm.25156 (no open licence) | rung M3b | **Table 1** and the protocol line of the text. Kept: T1 and single-echo T2 of the nine tubes; of the measured multi-echo T2 only the mean error per refocusing angle (which the table gives itself) and its spread |
| Hamilton 2011 | G. Hamilton and others, *NMR in Biomedicine* 24:784, 2011, doi:10.1002/nbm.1622 (no open licence) | row M4.1, M4.2 | **Table 3**: six observed positions and shares |
| Hernando 2014 | D. Hernando and others, *Magnetic Resonance in Medicine* 72:464, 2014, doi:10.1002/mrm.24951 (no open licence) | note of row M8.3 | one number: the slope of the fat-water shift against temperature |
| Hernando 2017 | D. Hernando and others, *Magnetic Resonance in Medicine* 77:1516, 2017, doi:10.1002/mrm.26228; data: doi:10.5281/zenodo.48266 (CC0) | rungs M4 and M8 | **the data** and its README; of the article, the four numbers of its abstract |
| Lu 2004 | H. Lu and others, *Magnetic Resonance in Medicine* 52:679, 2004, doi:10.1002/mrm.20178 | row M5.2 | the abstract only |
| IT'IS database | IT'IS Foundation, Tissue Properties Database, relaxation times, doi:10.13099/VIP21000-05-0 (licence unclear) | rung M5, with `--fetch` only; nothing is stored | the two spreadsheets as the page's table service returns them; which study gives which value is not in them |
| hazen | Guy's and St Thomas' NHS Foundation Trust, `GSTT-CSC/hazen` on GitHub, `tests/data/acr` (Apache-2.0, the licence of the repository) | rung M6 | **the 154 DICOM files** (pixels and the header fields a protocol needs) |
| Chen 2025 | Q. Chen and others, *Magnetic Resonance Materials in Physics, Biology and Medicine* 38(3):533, 2025, doi:10.1007/s10334-025-01247-1 (CC BY); data: doi:10.5281/zenodo.14778552 (CC BY) | rung M7 | **the data** (23 DICOM images) and the sentence of the article that gives T1 and T2 of the gel |
| Scholand 2023 | N. Scholand and others, *Magnetic Resonance in Medicine* 90:520, 2023, doi:10.1002/mrm.29664; data: doi:10.5281/zenodo.7654462 (CC BY) | rung M9 | **the raw data** (three of its files); of the article, Table 1 and the Methods paragraph on the phantom in the preprint arXiv:2209.08027; the authors' scripts for the B1 map and their reference values (`mrirecon/bloch-moba`) |

Journal, volume, first page and DOI of every article in this table agree with its Europe PMC record (read on
2026-11-26 and 2026-11-27); those of Chen 2025 with the citation fields of the publisher's page. None was checked
against Crossref.

## Not cited in the source

Statements that the package makes without a reference, so that a reader does not look for one:

- the Gibbs overshoot of 8.95 % (row K8), the Rayleigh moments 1.2533 and 0.6551 (row K16) and the null point
  T1 ln 2 (row S13) are standard results quoted as numbers;
- the proton gyromagnetic ratio 42.577478518 MHz/T carries no citation in the code;
- the sizes of the system errors and of the noise are choices of the package ([limits.md](limits.md), block 15).
