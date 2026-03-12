# Optical-Tweezers-Soft-Matter-
Senior Honours Physics project investigating complex fluid microrheology using optical tweezers and Python-based particle tracking analysis.

# Probing Complex Fluids with Optical Tweezers

The project investigates the viscoelastic response of complex fluids using a dual-trap optical tweezers setup. Micron-scale beads are trapped with focused laser beams and their motion is analysed to characterise the rheological properties of the surrounding fluid.

The experimental system was used to compare a Newtonian fluid (deionised water) with a viscoelastic polymer solution (polyacrylamide, PAM).

# Project Overview

Optical tweezers provide a microscopic probe for studying the mechanical response of soft materials. In this project, bead motion was tracked from microscope videos and analysed to extract rheological information.

Two complementary measurements were used:

- **Thermal fluctuations in a single optical trap**  
  Used to calculate trap stiffness and the high-frequency response of the fluid.

- **Trap flipping experiments**  
  Used to measure bead relaxation between two traps and probe the low-frequency viscoelastic response.

From these measurements, the storage and loss moduli of the fluid can be reconstructed across a range of frequencies.

---


### Code

The repository includes Python scripts used to analyse bead tracking data:

**Autocorrelation.py**

Computes the normalised position autocorrelation function from particle tracking data using an FFT-based method. The averaged autocorrelation can be fitted with an exponential decay to estimate relaxation times. 

**Normalised Position.py**

Processes trap-flipping experiments by detecting bead position steps, extracting trajectories around the flip event, normalising the displacement, and averaging multiple measurements to obtain the mean relaxation curve. 

**Position during bead flip.py**

Loads TrackMate tracking data, converts pixel coordinates to physical units, identifies trap centres using a Gaussian mixture model, and visualises bead motion during trap flipping experiments.

---

# Data

The raw experimental datasets are not included in this repository due to their size and formatting.

The scripts provided were originally used to analyse bead tracking data exported from ImageJ TrackMate.

---

# Viscoelastic Modulus Analysis

The code used to reconstruct the frequency-dependent storage and loss moduli from the combined stationary-trap and trap-flipping measurements is not included in this repository.

That analysis was performed using a separate in-house program used within the laboratory.

---

# Tools Used

Python libraries used in the analysis include:

- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn

---

# Author

Blazej Migo  
