# Research basis and scope

DriftForge uses public sources to set mechanisms and broad parameter families. The numerical ranges are domain-randomization hypotheses, not claims about one proprietary process node. They must be retuned if sponsor examples become available.

## Problem definition

- [SEMICON India 2026 hackathon — official challenge page](https://i4c.in/hackathon-2026/): Drift-Sense inputs/outputs, 10× scale relationship, DRAM/FinFET, independent noise, edge brightening, variation and centre tie rule.
- [Applied Materials official starter space](https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data/tree/main): compared for required interface and acquisition categories. DriftForge is original code and does not copy the starter.

## SEM image formation and distortion

- [NIST — Simulated SEM Images for Resolution Measurement](https://www.nist.gov/publications/simulated-sem-images-resolution-measurement): SEM simulation includes edge effects, substrate roughness, focus, drift/vibration and noise.
- [NIST — Real-Time Image Composition for Correction of Drift Distortion](https://www.nist.gov/publications/real-time-image-composition-correction-drift-distortion): raster-scan drift and distortion motivate row-wise jitter/shear.
- [NIST — Modeling and Simulation of Electron Microscopy](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908153): physical grounding for electron-image simulation and signal formation.

## DRAM and FinFET structural priors

- [6F² DRAM cell patent](https://patents.google.com/patent/US20020130348A1/en) and [6F² DRAM architecture paper](https://bioee.ee.columbia.edu/courses/ee6321/papers/00962294.pdf): repeated word/bit-line and contact organization motivates orthogonal periodic families and array/periphery boundaries.
- [UC Berkeley FinFET short course](https://people.eecs.berkeley.edu/~tking/presentations/KingLiu_2012VLSI-Tshortcourse.pdf): fin/gate geometry and pitch scaling motivate parallel fins crossed by gates.
- [FinFET electron-tomography study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6691105/): real FinFET morphology and nanoscale structural variation motivate non-ideal geometry.

## Registration and ambiguity

- [Guizar-Sicairos et al. — Efficient subpixel image registration algorithms](https://opg.optica.org/ol/fulltext.cfm?uri=ol-33-2-156): subpixel refinement.
- [Reddy & Chatterji — Fourier-Mellin rotation/scale/translation registration](https://dev.ipol.im/~reyotero/bib/bib_all/1996_Reddy_Chatterji_fft_based_trans_rot_scale_invar_registr.pdf): transform estimation alternative.
- [Doubek et al. — Image matching across repetitive patterns](https://cmp.felk.cvut.cz/~chum/papers/Doubek-ICPR10.pdf): repeated structures require explicit ambiguity reasoning.
- [Lewis — Fast Normalized Cross-Correlation](https://www.ncorr.com/download/publications/lewisfast.pdf): mathematical basis of the included audit baseline.

## Known limitations

- It is a 2D procedural proxy, not a Monte Carlo electron-transport simulator.
- Geometry families intentionally cover both DRAM-like and FinFET-like layouts but cannot reproduce undisclosed sponsor imagery exactly.
- Charging, PSF and detector-noise models are phenomenological.
- The structural signature distribution is a controllable training prior; stress/ambiguous profiles deliberately remove or weaken uniqueness.

