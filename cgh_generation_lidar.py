# =============================================================================
# CGH_generation_UOFA.py
#
# PLM (TI DLP6750-class) blazed-grating pattern generator.
#
# Geometry convention
# -------------------
#   i = row index (increases DOWNWARD), j = column index (increases RIGHT)
#   physical (x, y) = (j - cx, -(i - cy))          -> y points UP
#   steering angle theta measured CCW from +x
#   ramp coordinate  v = x*cos(t) + y*sin(t) = (j-cx)*cos(t) - (i-cy)*sin(t)
#   ideal blaze      psi(i,j) = 2*pi*v/Lambda + phi0   (Lambda in PLM phase pixels)
#
# Quantisation  (selected by QUANT_RULE)
# ------------
#   "nearest" : full-2*pi ramp + circular NEAREST-NEIGHBOUR search over the measured
#               level phases; dead-zone phases fold to the closer of P15 / P0.
#   "clip"    : same resetting sawtooth, but it SATURATES at P15 instead of folding.
#   "lut"     : THRESHOLDS_PCT used literally as decision boundaries on ramp position.
#
#   The first two replace the old "ramp scaled to phi_max + ceiling threshold" rule,
#   which rounded every cell UP to the next code and capped the achievable blaze at
#   phi_max instead of 2*pi.
#
# Guaranteed symmetry (verified at runtime, both anchor modes):
#   the level sequence along the grating vector for theta+180 is the exact
#   reverse of the sequence for theta.
# =============================================================================


import csv
import os


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TWO_PI = 2.0 * np.pi




# =============================================================================
#  RUN CONFIGURATION  --  everything you normally touch lives in this block
# =============================================================================


# -----------------------------------------------------------------------------
# 1. PHASE ANCHORING  --  uncomment exactly ONE
# -----------------------------------------------------------------------------
#   "free" : a global phase offset is swept and the best one is used.  A constant
#            phase across the whole panel is invisible in the far field, so this
#            is FREE efficiency.  Ramp origin sits at the array centre and the
#            absolute codes will NOT match the lab convention
#            (Lambda=2 -> P7/P15 @ 99.5 %, Lambda=3 -> P0/P12/P15 @ 99.5 %).
#
#   "P0"   : the ramp's zero-phase reference is pinned to code P0, matching the lab
#            convention.  The reference sits at the array CENTRE (not the corner) so
#            that the panel window stays symmetric about it and theta / theta+180
#            give identical efficiency.  Costs a few points versus "free".
#


PHASE_ANCHOR = "free"


ANCHOR_LEVEL = 0          # code pinned to the ramp origin when PHASE_ANCHOR == "P0"




# -----------------------------------------------------------------------------
# 1b. QUANTISER RULE  --  uncomment exactly ONE
# -----------------------------------------------------------------------------
#   "clip"    : TRUNCATED blaze.  The sawtooth still RESETS every period (integer or
#               not), but within a period the ramp keeps its true 2*pi slope and
#               SATURATES at the top code: phases above P15 that the device cannot
#               reach are held at P15 instead of folding back down to P0.
#               More codes get used at long and non-integer periods.  Intended mode.
#
#   "nearest" : same resetting sawtooth, but phases in the unreachable dead zone
#               (1.31 pi -> 2 pi) snap to whichever of P15 / P0 is circularly
#               closer, so the top of each period folds early back to P0.
#               Higher efficiency than "clip", at the cost of that extra fold.
#
#   "lut"     : use THRESHOLDS_PCT literally as decision boundaries on the ramp
#               POSITION n (0 -> 100 % across one period):
#                   n < 1.374            -> P0
#                   1.374 <= n < 2.140   -> P1   ... etc
#               This is the bench/lab rule and what the original script did.
#               PHASE_ANCHOR is ignored (the LUT is anchored to the ramp itself).
#
QUANT_RULE = "lut"
# QUANT_RULE = "lut"
# QUANT_RULE = "clip"




# -----------------------------------------------------------------------------
# 1c. LEVEL PHASE MODEL  --  what does THRESHOLDS_PCT = 100 % mean in PHASE?
#     This does NOT change the LUT patterns, but it decides every efficiency
#     number and every "nearest" decision, so it has to be right.
# -----------------------------------------------------------------------------
#   "piston"  : code % is a fraction of MAX_DISPLACEMENT_NM.
#               phi = 4*pi*d/lambda  ->  full scale = 1.3094 pi at 905 nm.
#               The blaze can never wrap, so efficiency is capped near 84 %.
#
#   "full2pi" : code % is a fraction of a full 2*pi wrap (what a ramp LUT
#               normally implies).  full scale = 2 pi, and MAX_DISPLACEMENT_NM
#               must then really be lambda/2 = 452.5 nm, not 296.25 nm.
#               Under this model P0/P13 at Lambda=2 is a good pair (98.2 %).
#
LEVEL_PHASE_MODEL = "piston"
# LEVEL_PHASE_MODEL = "full2pi"




# -----------------------------------------------------------------------------
# 2. STEERING ANGLES  --  uncomment exactly ONE (or write your own list)
# -----------------------------------------------------------------------------
ANGLES_DEG = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]      # 8 angles, 45 deg step
# ANGLES_DEG = [0.0, 90.0, 180.0, 270.0]                               # cardinal only
# ANGLES_DEG = list(np.arange(0.0, 360.0, 22.5))                       # 16 angles, 22.5 deg step
# ANGLES_DEG = list(np.arange(0.0, 360.0, 10.0))                       # 36 angles, 10 deg step


FLAT_ANGLES = set()   # e.g. {0.0} emits 0 deg as a flat state instead of a grating




# -----------------------------------------------------------------------------
# 3. GRATING PERIODS [PLM pixels]  --  uncomment exactly ONE
#    Lambda < 2 violates Nyquist: it aliases and steers the wrong way.
# -----------------------------------------------------------------------------
PERIODS = np.round(np.arange(1.0, 3.01, 0.1), 2)      # your original sweep (includes aliased)
# PERIODS = np.round(np.arange(2.0, 3.01, 0.1), 2)    # Nyquist-safe subset of the above
# PERIODS = np.round(np.arange(2.0, 8.01, 0.5), 2)    # wider / shallower steering
# PERIODS = [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]            # sparse spot check




# -----------------------------------------------------------------------------
# 4. OUTPUT TOGGLES
# -----------------------------------------------------------------------------
# fmt: off
SAVE_BMP          = True   # write one full-resolution CGH bitmap per angle x period
SAVE_TILE_BMP     = True   # write the sixteen 20x20 tile-key bitmaps


DERIVE_COMPLEMENTS = True   # compute only theta < 180 and obtain theta+180 as rot180 of it.
                            # rot180 of a valid theta grating IS a valid theta+180 grating
                            # (it differs from the recomputed one by a constant phase, which
                            # is invisible in the far field), so the complementary codes are
                            # exact flips by construction and the efficiency is identical.
CSV_CUTS          = True    # dump a horizontal AND a vertical cut of codes per angle x period.
                            # For theta = 90/270 the grating runs vertically, so a ROW is
                            # constant and only the COLUMN shows the code sequence.
CUT_LEN           = 25      # samples per cut (odd, centred on the array centre)
PRINT_MATRIX      = True    # console: print the NxN code matrix, LOG_LAMBDAS periods only
CSV_MATRIX        = True    # CSV: write the NxN code matrix for EVERY angle x period
MATRIX_SIZE       = 20      # N, the logged block size
LOG_LAMBDAS       = [1.0, 1.4, 2.0, 2.1, 2.5, 3.0]   # which periods get printed to console


PLOT_FAR_FIELD    = "all"   # "off"   -> no far-field plots
                            # "cases" -> only SPECTRUM_CASES below
                            # "all"   -> one plot per angle x period (SLOW, many files)


PLOT_GRID_COMPOSITE = True  # stack the GRID_SPOTS far fields into ONE image so every
                            # steered spot of the grid is visible at once, each target
                            # order circled and labelled with its efficiency
GRID_SPECTRUM_N     = 512   # crop size for the composite (bigger = sharper spots)
SPECTRUM_WINDOW     = False # apply a 2-D Hann window before the display FFT.
                            # A hard-edged crop convolves EVERY spot with the same sinc
                            # cross, which makes the spots look identical and stamped.
                            # Display only -- all eta numbers are analytic, not FFT-derived.
SPECTRUM_ENVELOPE   = True  # multiply the displayed spectrum by the square-mirror element
                            # pattern sinc^2(f). WITHOUT this the plot is array-factor only
                            # (every pixel a point emitter) and all spots look equally
                            # bright; with it the relative brightness is what a camera in
                            # the Fourier plane would actually record.


# The steering grid. (label, theta_deg, period_px); period None = flat / undiffracted.
# The corners MUST use GRID_EDGE/sqrt(2) for a true square grid -- that puts them at the
# same f on both axes as the edges (0.47619 for GRID_EDGE = 2.1). A rounded 1.5 lands at
# 0.47140, ~1 % off grid, and costs ~11 points of spot power.
GRID_EDGE   = 2.1
GRID_EDGE   = 2.1                       # edge (E/N/W/S) period in PLM pixels
GRID_CORNER = GRID_EDGE / np.sqrt(2)    # 1.48492 -- do not round this


GRID_SPOTS = [
    ("centre", None,  None),
    ("E",   0.0, GRID_EDGE),   ("N",   90.0, GRID_EDGE),
    ("W", 180.0, GRID_EDGE),   ("S",  270.0, GRID_EDGE),
    ("NE",  45.0, GRID_CORNER), ("NW", 135.0, GRID_CORNER),
    ("SW", 225.0, GRID_CORNER), ("SE", 315.0, GRID_CORNER),
]
PLOT_PHASE_RAMP   = "all"
                            # "cases" -> overview grid only (one panel per period)
                            # "all"   -> overview grid + one detailed plot per period
SPECTRUM_CASES    = [(0.0, 2.0), (0.0, 3.0), (45.0, 2.5), (180.0, 3.0)]
SPECTRUM_N        = 256     # centred crop size used for the far-field FFT
# fmt: on




# ========================== DEVICE / OPTICS CONFIG ==========================
# fmt: off
PLM_WIDTH           = 2716        # DMD-format bitmap width  (= 2 x phase pixels)
PLM_HEIGHT          = 1600        # DMD-format bitmap height (= 2 x phase pixels)
PIXEL_PITCH_UM      = 10.8        # phase-pixel pitch
FILL_FACTOR         = 0.92         # LINEAR fill factor of the piston mirror (1.0 = ideal)


LAMBDA_NM           = 905.0       # operating wavelength
MAX_DISPLACEMENT_NM = 296.25      # piston travel at code = 100 %


N_OFFSETS           = 360         # offset search resolution ("free" anchor only)
N_OFFSET_BINS       = 2048        # histogram resolution for the offset search


FLAT_PHASE_LEVEL    = 0           # tile used for the "flat" state
# fmt: on


# ============================ CONFIG VALIDATION ============================
# Every switch above is a bare string that falls through to a default when misspelled,
# and the output folder name is built from three of them -- so a typo would silently
# produce a confidently mislabelled dataset. Fail loudly instead.
# fmt: off
_VALID_CHOICES = {
    "PHASE_ANCHOR"      : ("free", "P0"),
    "QUANT_RULE"        : ("nearest", "clip", "lut"),
    "LEVEL_PHASE_MODEL" : ("piston", "full2pi"),
    "PLOT_FAR_FIELD"    : ("off", "cases", "all"),
    "PLOT_PHASE_RAMP"   : ("off", "cases", "all"),
}
# fmt: on
for _name, _allowed in _VALID_CHOICES.items():
    if globals()[_name] not in _allowed:
        raise ValueError(f"{_name} = {globals()[_name]!r} is not one of {_allowed}")
if not 0 <= ANCHOR_LEVEL <= 15:
    raise ValueError(f"ANCHOR_LEVEL = {ANCHOR_LEVEL} must be a code index in 0..15")
if len(PERIODS) == 0 or len(ANGLES_DEG) == 0:
    raise ValueError("PERIODS and ANGLES_DEG must both be non-empty")


# derived from QUANT_RULE -- do not edit
#   The ramp origin must sit at the array CENTRE, otherwise the panel window is not
#   symmetric about it: theta and theta+180 then see different windows of the same
#   infinite pattern and their efficiencies differ by up to 10 points at low-denominator
#   periods (e.g. Lambda = 2.2 = 11/5).  "lut" keeps the corner to match the bench.
ORIGIN = "corner" if QUANT_RULE == "lut" else "center"




# ========================== PLM PHASE CALIBRATION ==========================
# Piston displacement of code Pk, as a FRACTION (%) of MAX_DISPLACEMENT_NM.
# If you have a directly measured phase-vs-code curve, put it in
# PHASE_LEVELS_RAD_OVERRIDE and every downstream stage adapts automatically.
# fmt: off
THRESHOLDS_PCT = np.array([
     1.37420442,  2.140099448,  3.339651934,  4.665104972,
     5.80170718,  7.928027624, 11.687558010, 20.252292820,
    27.61397790, 30.604116020, 35.112055250, 41.697353590,
    48.24418232, 55.619856350, 65.705889500, 100.0
])
# fmt: on


PHASE_LEVELS_RAD_OVERRIDE = None  # e.g. np.array([...]) of 16 measured phases in radians




# ========================== 2x2 BINARY BIT TILES ==========================
# 4-bit hardware code for phase state P0..P15, written as 2x2 sub-pixels.
# fmt: off
BIT_TILES = np.array([
    [[0, 1], [0, 1]],   # P0
    [[0, 1], [0, 0]],   # P1
    [[0, 0], [0, 1]],   # P2
    [[0, 1], [1, 1]],   # P3
    [[0, 0], [0, 0]],   # P4
    [[0, 1], [1, 0]],   # P5
    [[0, 0], [1, 1]],   # P6
    [[0, 0], [1, 0]],   # P7
    [[1, 1], [0, 1]],   # P8
    [[1, 1], [0, 0]],   # P9
    [[1, 0], [0, 1]],   # P10
    [[1, 0], [0, 0]],   # P11
    [[1, 1], [1, 1]],   # P12
    [[1, 1], [1, 0]],   # P13
    [[1, 0], [1, 1]],   # P14
    [[1, 0], [1, 0]],   # P15
], dtype=np.uint8)
# fmt: on




# ============================ PHASE LEVEL TABLE ============================
class PhaseLevelTable:
    """Nearest-neighbour quantiser over a non-uniform, circular (mod 2*pi) level set."""


    def __init__(self, phases_rad):
        self.phase = np.mod(np.asarray(phases_rad, dtype=np.float64), TWO_PI)
        self.n = self.phase.size
        self.order = np.argsort(self.phase)                 # sorted position -> level index
        srt = self.phase[self.order]
        self.sorted_phase = srt
        self.phi_min, self.phi_max = float(srt[0]), float(srt[-1])
        self._mid = 0.5 * (srt[:-1] + srt[1:])              # non-circular decision boundaries
        # replicate the level set one period below and above so the search wraps correctly
        self._ext = np.concatenate([srt - TWO_PI, srt, srt + TWO_PI])
        self._edges = 0.5 * (self._ext[:-1] + self._ext[1:])


    def quantize(self, psi):
        """psi (any shape, radians) -> level index whose phase is circularly closest."""
        k = np.searchsorted(self._edges, np.mod(psi, TWO_PI), side="left")
        return self.order[np.mod(k, self.n)]


    def quantize_clipped(self, psi):
        """Nearest level WITHOUT wrap-around: anything above the top code saturates there
        instead of folding back to the bottom code. Used by the truncated-ramp mode."""
        k = np.searchsorted(self._mid, np.clip(psi, self.phi_min, self.phi_max), side="left")
        return self.order[np.clip(k, 0, self.n - 1)]


    def realized(self, idx):
        return self.phase[idx]


    def max_quantization_error(self):
        srt = np.sort(self.phase)
        gaps = np.diff(np.concatenate([srt, [srt[0] + TWO_PI]]))
        return 0.5 * gaps.max()




def build_phase_levels():
    """Returns (displacement_nm, phase_rad, PhaseLevelTable) for the 16 codes."""
    if PHASE_LEVELS_RAD_OVERRIDE is not None:
        raw = np.asarray(PHASE_LEVELS_RAD_OVERRIDE, dtype=np.float64)
        disp = raw * LAMBDA_NM / (4.0 * np.pi)
    elif LEVEL_PHASE_MODEL == "full2pi":
        raw = TWO_PI * THRESHOLDS_PCT / 100.0
        disp = raw * LAMBDA_NM / (4.0 * np.pi)
    else:
        disp = THRESHOLDS_PCT / 100.0 * MAX_DISPLACEMENT_NM
        raw = 4.0 * np.pi * disp / LAMBDA_NM          # reflective double pass
    return disp, raw, PhaseLevelTable(raw)




# ============================== GEOMETRY ==============================
def direction_cosines(theta_deg, tol=1e-12):
    """cos/sin with cardinal-angle residues snapped to exact 0 / +-1, and angles in
    [180, 360) derived by explicit negation of their complement.


    Two reasons: sin(180 deg) evaluates to 1.22e-16 rather than 0, which makes any
    threshold rule flip whole stripes at the sawtooth wrap; and cos(225 deg) is not
    bit-identical to -cos(45 deg), which leaves a ~1e-3 efficiency gap between
    complementary diagonal angles.  Both are removed here."""
    t = float(theta_deg) % 360.0
    flip = t >= 180.0
    if flip:
        t -= 180.0
    r = np.deg2rad(t)
    c, s = np.cos(r), np.sin(r)
    c = 0.0 if abs(c) < tol else c
    s = 0.0 if abs(s) < tol else s
    return (-c, -s) if flip else (c, s)




def fold_to_first_zone(f):
    """Fold a spatial frequency into the sampled first zone [-0.5, 0.5).


    f - round(f) is WRONG at the zone edge: numpy rounds half to even, so f = +0.5 maps
    to itself, but the sampled grid only carries -0.5 (fftshift bin 0). The target marker
    then lands on the opposite edge of the plot from the spot it is supposed to circle."""
    return np.mod(f + 0.5, 1.0) - 0.5




def freq_extent(n):
    """imshow extent for an fftshift-ed n x n spectrum.


    extent gives the OUTER edges of the image, not the bin centres, so it has to be the
    frequency axis widened by half a bin -- otherwise every analytic marker sits half a
    bin away from the data it annotates."""
    f = np.fft.fftshift(np.fft.fftfreq(n))
    h = 0.5 / n
    return [f[0] - h, f[-1] + h, f[0] - h, f[-1] + h]




def ramp_coordinate(rows, cols, theta_deg, origin=None):
    """v(i, j) = (j - cx) cos(t) - (i - cy) sin(t)   [units: phase pixels]"""
    origin = ORIGIN if origin is None else origin
    if origin == "center":
        cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    else:
        cy, cx = 0.0, 0.0
    ct, st = direction_cosines(theta_deg)
    i = (np.arange(rows, dtype=np.float64) - cy)[:, None]
    j = (np.arange(cols, dtype=np.float64) - cx)[None, :]
    return j * ct - i * st




def make_index_map_lut(v, period):
    """Lab LUT rule: THRESHOLDS_PCT are decision boundaries on the ramp POSITION n,
    where n sweeps 0 -> 100 % across one grating period.


        n <  THRESHOLDS_PCT[0]                      -> P0
        THRESHOLDS_PCT[k-1] <= n < THRESHOLDS_PCT[k] -> Pk


    This is the rule the original script implemented and the one used on the bench."""
    n = np.mod(100.0 * v / period, 100.0)
    k = np.searchsorted(THRESHOLDS_PCT, n, side="right")
    return np.clip(k, 0, THRESHOLDS_PCT.size - 1).astype(np.uint8)




def ramp_phase(v, period, table, shift=0.0):
    """Sawtooth phase for the active rule.


    'shift' is a LATERAL translation of the grating in fractions of a period, not a
    phase offset.  For a truncated ramp the two are NOT equivalent: without wrap-around
    a negative phase offset cannot be reached by adding 2*pi, so sweeping a phase offset
    would only ever push the whole ramp up into saturation.  A lateral shift is free in
    the far field and is the correct degree of freedom."""
    psi = TWO_PI * np.mod(v / period + shift, 1.0)
    return psi + table.phi_min if QUANT_RULE == "clip" else psi




def make_index_map(v, period, table, shift=0.0):
    """Wrapped blaze + circular nearest-neighbour quantisation."""
    return table.quantize(TWO_PI * np.mod(v / period + shift, 1.0)).astype(np.uint8)




def make_index_map_clip(v, period, table, shift=0.0):
    """TRUNCATED blaze. The sawtooth RESETS every period via mod(v/period, 1) -- that is
    true for integer and non-integer periods alike. Within a period the ramp keeps its
    true 2*pi slope and saturates at the top code, so phases the device cannot reach are
    held at P15 rather than folding back to P0."""
    psi = TWO_PI * np.mod(v / period + shift, 1.0) + table.phi_min
    return table.quantize_clipped(psi).astype(np.uint8)




def build_index_map(v, period, table, shift=0.0):
    """Single dispatch point for QUANT_RULE."""
    if QUANT_RULE == "lut":
        return make_index_map_lut(v, period)
    if QUANT_RULE == "clip":
        return make_index_map_clip(v, period, table, shift)
    return make_index_map(v, period, table, shift)




# retained under the old name so existing call sites keep working
make_index_map_legacy = make_index_map_lut




def index_map_to_bitmap(idx_map, tiles=BIT_TILES):
    """Vectorised 2x2 tile expansion (replaces the 1.09 M-iteration Python loop)."""
    r, c = idx_map.shape
    return tiles[idx_map].transpose(0, 2, 1, 3).reshape(2 * r, 2 * c)




# ============================== EFFICIENCY ==============================
def grating_efficiency(idx_map, v, period, table):
    """Power fraction into the target plane wave: |<exp(i(phi_real - phi_ideal))>|^2.
    Exact for non-integer periods (unlike an FFT bin)."""
    err = table.realized(idx_map) - TWO_PI * v / period
    return float(np.cos(err).mean() ** 2 + np.sin(err).mean() ** 2)




def order_degeneracy(period, theta_deg, tol=1e-9):
    """Returns 2 when the -1 order lands on the SAME sampled frequency as the +1 order,
    otherwise 1.


    This happens when 2*f is an integer vector, i.e. the grating sits exactly at Nyquist
    (Lambda = 2 px along a cardinal axis).  Such a grating is its own mirror image: the
    code sequence for theta and theta+180 is identical and the power splits equally
    between the two first orders, so it CANNOT steer to one side only.  Unidirectional
    steering needs Lambda strictly greater than 2 px."""
    ct, st = direction_cosines(theta_deg)
    fx2, fy2 = 2.0 * ct / period, 2.0 * st / period
    degenerate = (abs(fx2 - round(fx2)) < tol) and (abs(fy2 - round(fy2)) < tol)
    return 2 if degenerate else 1




def pixel_envelope(period, theta_deg, fill=FILL_FACTOR):
    """Square-mirror aperture envelope at the target order (also caps eta at fill^2)."""
    ct, st = direction_cosines(theta_deg)
    fx, fy = ct / period, st / period
    return float(fill**2 * np.sinc(fill * fx) ** 2 * np.sinc(fill * fy) ** 2)




def quantize_for_rule(table, psi):
    """Apply whichever phase->code mapping the active QUANT_RULE uses.
    'clip' must NOT wrap psi, otherwise the saturation behaviour is lost."""
    if QUANT_RULE == "clip":
        return table.quantize_clipped(psi)
    return table.quantize(psi)




def optimise_shift(v, period, table, n_shifts=N_OFFSETS, n_bins=N_OFFSET_BINS):
    """Best lateral shift of the grating, in fractions of a period.


    A lateral shift does not change the far-field intensity of an ideal grating, but it
    does change which pixels land where on the ramp, and therefore which codes get used.
    With 16 non-uniformly spaced levels that is worth a few points. O(N) + O(n_shifts*n_bins).
    """
    u = np.mod(v / period, 1.0).ravel()
    b = np.minimum((u * n_bins).astype(np.int64), n_bins - 1)
    w = np.exp(-1j * TWO_PI * u)
    S = (np.bincount(b, weights=w.real, minlength=n_bins)
         + 1j * np.bincount(b, weights=w.imag, minlength=n_bins))


    u_c = (np.arange(n_bins) + 0.5) / n_bins
    shifts = np.arange(n_shifts) / n_shifts
    psi = TWO_PI * np.mod(u_c[None, :] + shifts[:, None], 1.0)
    if QUANT_RULE == "clip":
        psi = psi + table.phi_min
    phi = table.realized(quantize_for_rule(table, psi))
    eta = np.abs(np.exp(1j * phi) @ S) ** 2 / u.size**2
    return float(shifts[int(np.argmax(eta))])




def resolve_shift(theta, period, v, table, cache):
    """Shift used for a given (theta, Lambda). Cached per (theta mod 180, Lambda) so that
    complementary angles always share it and stay exact mirrors of each other.


    In 'clip' mode the ramp base is phi(P0) by construction, so the ramp always starts at
    P0 no matter what the shift is -- PHASE_ANCHOR is therefore irrelevant there and the
    shift is always optimised.  In 'nearest' mode PHASE_ANCHOR == "P0" pins the ramp
    origin to code P{ANCHOR_LEVEL} instead of optimising."""
    if QUANT_RULE == "lut":
        return 0.0
    if QUANT_RULE == "nearest" and PHASE_ANCHOR == "P0":
        # put the anchor code's phase exactly at the ramp origin, so the ramp really
        # starts on P{ANCHOR_LEVEL} rather than merely landing near it by accident
        return float(table.phase[ANCHOR_LEVEL] / TWO_PI)
    key = (round(theta % 180.0, 6), float(period))
    if key not in cache:
        cache[key] = optimise_shift(v, period, table)
    return cache[key]




def alias_info(period, theta_deg):
    """Sampled-grid aliasing. The Nyquist test is PER COMPONENT, not on 1/Lambda:
    a 45 deg grating at Lambda = 1.5 has fx = fy = 0.471 < 0.5 and is perfectly valid,
    while the same period at 0 deg has fx = 0.667 and aliases.  A square steering grid
    relies on exactly this -- its corner spots need Lambda_axial/sqrt(2)."""
    ct, st = direction_cosines(theta_deg)
    fx, fy = ct / period, st / period
    ok = abs(fx) <= 0.5 + 1e-12 and abs(fy) <= 0.5 + 1e-12
    fxa, fya = fold_to_first_zone(fx), fold_to_first_zone(fy)
    mag = float(np.hypot(fxa, fya))
    eff_period = np.inf if mag == 0 else 1.0 / mag
    # at the zone edge the +1 and -1 orders coincide, so "reversed" carries no meaning
    rev = False if order_degeneracy(period, theta_deg) == 2 else bool(fxa * fx + fya * fy < 0)
    return ok, eff_period, rev




def steering_deg(period):
    s = (LAMBDA_NM * 1e-3) / (period * PIXEL_PITCH_UM)
    return float(np.degrees(np.arcsin(s))) if abs(s) <= 1.0 else float("nan")




# ============================= SYMMETRY TEST =============================
def symmetry_report(table, rows, cols, cache, half=400):
    """Anchor-independent: the level sequence along the grating vector for theta+180
    must be the exact reverse of the sequence for theta.


    rows/cols and the shared shift cache come from the caller so the test uses the SAME
    shifts as the patterns that actually get written -- optimising the shift on a
    different array size would validate a pattern that is never emitted."""
    lines, worst = [], 0
    v_line = np.arange(-half, half + 1, dtype=np.float64)
    for theta in [0.0, 45.0, 90.0, 135.0]:
        v_full = ramp_coordinate(rows, cols, theta)
        for period in [2.0, 2.3, 3.0]:
            off = resolve_shift(theta, period, v_full, table, cache)
            fwd = build_index_map(v_line[None, :], period, table, off)[0]
            rev = build_index_map(-v_line[None, :], period, table, off)[0]
            bad = int(np.count_nonzero(rev != fwd[::-1]))
            worst = max(worst, bad)
            lines.append(f"  theta={theta:6.1f}  L={period:4.2f}   "
                         f"reversed-sequence mismatches = {bad}")
    return lines, worst




# ================================ PLOTS ================================
def plot_calibration(disp_nm, phase_raw, table, outdir):
    fig = plt.figure(figsize=(13, 5), dpi=150)
    fig.suptitle("PLM phase-level calibration", fontweight="bold")


    ax = fig.add_subplot(1, 3, 1)
    ax.plot(np.arange(16), phase_raw / np.pi, "o-", label="level phase")
    ax.axhline(2.0, ls="--", c="r", label="2$\\pi$ (needed for 100 %)")
    ax.set_xlabel("phase code Pk"); ax.set_ylabel("phase [$\\pi$ rad]")
    ax.set_title(f"span = {phase_raw.max()/np.pi:.3f}$\\pi$ @ {LAMBDA_NM:.0f} nm")
    ax.grid(alpha=.3); ax.legend(fontsize=8)


    ax = fig.add_subplot(1, 3, 2)
    ax.plot(np.arange(16), disp_nm, "s-", c="g")
    ax.set_xlabel("phase code Pk"); ax.set_ylabel("piston displacement [nm]")
    ax.set_title("displacement"); ax.grid(alpha=.3)


    ax = fig.add_subplot(1, 3, 3, projection="polar")
    srt = np.sort(table.phase)
    ax.plot(srt, np.ones_like(srt), "o", ms=7)
    for p in srt:
        ax.plot([p, p], [0, 1], lw=.8, c="0.7")
    ax.set_yticklabels([])
    ax.set_title(f"level coverage on the unit circle\nmax quant. error = "
                 f"{table.max_quantization_error():.3f} rad", fontsize=9)


    fig.tight_layout()
    path = os.path.join(outdir, "Phase_Level_Calibration.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    return path




def plot_tile_key(phase_raw, outdir):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=150)
    fig.suptitle("20x20 sub-pixel binary pattern key (P0 - P15)", fontsize=14, fontweight="bold")
    for p in range(16):
        ax = axes[p // 4, p % 4]
        tile20 = np.tile(BIT_TILES[p], (10, 10))
        ax.imshow(tile20, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"P{p} ({phase_raw[p]/np.pi:.2f}$\\pi$)", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if SAVE_TILE_BMP:
            Image.fromarray(np.uint8(tile20 * 255)).save(os.path.join(outdir, f"Tile_P{p}_20x20.bmp"))
    fig.tight_layout()
    path = os.path.join(outdir, "Phase_Level_Tiles_P0_P15_Key.png")
    fig.savefig(path, dpi=200); plt.close(fig)
    return path




def plot_efficiency(records, outdir):
    angles = sorted({r["theta"] for r in records})
    periods = sorted({r["period"] for r in records})
    shape = (len(angles), len(periods))
    g, t, fr, lg = (np.full(shape, np.nan) for _ in range(4))
    for r in records:
        a, p = angles.index(r["theta"]), periods.index(r["period"])
        g[a, p], t[a, p] = r["eta_grating"], r["eta_total"]
        fr[a, p], lg[a, p] = r["eta_free"], r["eta_legacy"]


    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle(f"Diffraction efficiency into the +1 (target) order   "
                 f"[PHASE_ANCHOR = '{PHASE_ANCHOR}']", fontweight="bold")
    for k, a in enumerate(angles):
        axes[0].plot(periods, 100 * g[k], "o-", ms=3, label=f"{a:g}$\\degree$")
        axes[1].plot(periods, 100 * t[k], "o-", ms=3, label=f"{a:g}$\\degree$")
    axes[0].set_title("grating (phase) efficiency")
    axes[1].set_title(f"incl. square-mirror envelope (fill = {FILL_FACTOR:g})")
    for ax in axes:
        ax.axvline(2.0, ls="--", c="r", lw=1)
        ax.text(2.02, 3, "Nyquist limit", color="r", fontsize=8, rotation=90, va="bottom")
        ax.set_xlabel("grating period $\\Lambda$ [PLM pixels]")
        ax.set_ylabel("efficiency [%]"); ax.grid(alpha=.3); ax.set_ylim(0, 105)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p1 = os.path.join(outdir, "Diffraction_Efficiency_vs_Period.png")
    fig.savefig(p1, dpi=200); plt.close(fig)


    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    im = ax.imshow(100 * g, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100,
                   extent=[periods[0] - .05, periods[-1] + .05, -0.5, len(angles) - 0.5])
    ax.set_yticks(range(len(angles))); ax.set_yticklabels([f"{a:g}$\\degree$" for a in angles])
    ax.set_xlabel("grating period $\\Lambda$ [PLM pixels]"); ax.set_ylabel("steering angle")
    ax.set_title("Grating efficiency $\\eta$ [%]", fontweight="bold")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    p2 = os.path.join(outdir, "Diffraction_Efficiency_Heatmap.png")
    fig.savefig(p2, dpi=200); plt.close(fig)


    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    ax.plot(periods, 100 * np.nanmean(lg, axis=0), "s--", c="crimson",
            label="original: $\\phi_{max}$-scaled ramp + ceiling")
    ax.plot(periods, 100 * np.nanmean(g, axis=0), "o-", c="seagreen",
            label=f"this run: nearest, anchor = '{PHASE_ANCHOR}'")
    if PHASE_ANCHOR != "free":
        ax.plot(periods, 100 * np.nanmean(fr, axis=0), "^:", c="steelblue",
                label="nearest, anchor = 'free' (upper bound)")
    ax.axvline(2.0, ls="--", c="r", lw=1)
    ax.set_xlabel("grating period $\\Lambda$ [PLM pixels]")
    ax.set_ylabel("grating efficiency, angle-averaged [%]")
    ax.set_title("Quantiser comparison", fontweight="bold")
    ax.grid(alpha=.3); ax.set_ylim(0, 105); ax.legend(fontsize=9)
    fig.tight_layout()
    p3 = os.path.join(outdir, "Method_Comparison.png")
    fig.savefig(p3, dpi=200); plt.close(fig)
    return p1, p2, p3




def _ramp_profile(period, table, shift, n_periods=2):
    """Sampled pixel positions over n_periods, with the ideal and realised phase."""
    n_px = max(int(np.ceil(n_periods * period)) + 1, 6)
    x = np.arange(n_px, dtype=np.float64)
    idx = build_index_map(x[None, :], period, table, shift)[0]
    realised = table.realized(idx)
    demanded = ramp_phase(x, period, table, shift)     # what the blaze asks for
    return x, idx, realised, demanded




# What happens to a sample whose demanded phase sits above P15 depends on the rule:
# only 'clip' truncates. 'nearest' folds it to the circularly closer of P15 / P0 --
# usually P0 -- and 'lut' never builds a phase ramp, so nothing is unreachable.
# fmt: off
_DEAD_ZONE_LABEL = {"clip": "truncated at P15", "nearest": "dead zone -> folded"}
_DEAD_ZONE_WORD  = {"clip": "truncated", "nearest": "folded", "lut": "n/a"}
# fmt: on


def _dead_zone_mask(demanded, table):
    """Samples the device cannot reach. Always empty for 'lut', which has no phase ramp."""
    if QUANT_RULE == "lut":
        return np.zeros(np.shape(demanded), dtype=bool)
    return demanded > table.phi_max




def _draw_ramp(ax, period, table, shift, show_levels=True):
    x, idx, realised, demanded = _ramp_profile(period, table, shift)
    t = np.linspace(0, x[-1], 2000)
    demand_t = ramp_phase(t, period, table, shift)


    # region the device physically cannot reach
    ax.axhspan(table.phi_max / np.pi, 2.05, color="crimson", alpha=.08, zorder=0)
    ax.plot(t, demand_t / np.pi, "--", c="0.55", lw=1, label="demanded blaze")
    ax.step(x, realised / np.pi, where="mid", c="seagreen", lw=1.8, label="realised")


    flagged = _dead_zone_mask(demanded, table)
    ax.plot(x[~flagged], realised[~flagged] / np.pi, "o", c="seagreen", ms=4)
    if flagged.any():
        ax.plot(x[flagged], realised[flagged] / np.pi, "o", c="crimson", ms=5,
                label=_DEAD_ZONE_LABEL.get(QUANT_RULE, "unreachable"))
    ax.axhline(table.phi_max / np.pi, c="crimson", ls=":", lw=1.2,
               label=f"P15 = {table.phi_max/np.pi:.3f}$\\pi$")
    if show_levels:
        for p in table.sorted_phase:
            ax.axhline(p / np.pi, c="0.9", lw=.6, zorder=0)
    ax.set_ylim(-0.05, 2.05)
    ax.set_xlim(0, x[-1])
    return idx, flagged




def plot_phase_ramps(table, outdir, shift_for):
    if PLOT_PHASE_RAMP == "off":
        return []
    periods = [float(p) for p in PERIODS]
    paths = []


    ncol = 4
    nrow = int(np.ceil(len(periods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.5 * nrow), dpi=130,
                             squeeze=False, sharey=True)
    fig.suptitle(f"Phase ramp actually written to the PLM   [QUANT_RULE = '{QUANT_RULE}', "
                 f"{LEVEL_PHASE_MODEL}]", fontweight="bold")
    for k, period in enumerate(periods):
        ax = axes[k // ncol, k % ncol]
        idx, flagged = _draw_ramp(ax, period, table, shift_for(period), show_levels=False)
        ax.set_title(f"$\\Lambda$ = {period:g} px   ({len(np.unique(idx))} codes, "
                     f"{100*flagged.mean():.0f} % {_DEAD_ZONE_WORD[QUANT_RULE]})", fontsize=9)
        ax.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("phase [$\\pi$ rad]", fontsize=8)
    for k in range(len(periods), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    p = os.path.join(outdir, "Phase_Ramp_Overview.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    paths.append(p)


    if PLOT_PHASE_RAMP == "all":
        for period in periods:
            fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
            idx, flagged = _draw_ramp(ax, period, table, shift_for(period))
            codes = " ".join(f"P{k}" for k in idx[:min(len(idx), 12)])
            ax.set_xlabel("position along grating vector [PLM pixels]")
            ax.set_ylabel("phase [$\\pi$ rad]")
            ax.set_title(f"Phase ramp   $\\Lambda$ = {period:g} px   [{QUANT_RULE}]   "
                         f"{100*flagged.mean():.0f} % of samples "
                         f"{_DEAD_ZONE_WORD[QUANT_RULE]}\ncodes: {codes}",
                         fontweight="bold", fontsize=10)
            ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower right")
            fig.tight_layout()
            q = os.path.join(outdir, "phase_ramp", f"PhaseRamp_L{period:g}.png")
            os.makedirs(os.path.dirname(q), exist_ok=True)
            fig.savefig(q, dpi=160); plt.close(fig)
            paths.append(q)
    return paths




def element_pattern(n, fill=FILL_FACTOR):
    """Square-mirror element pattern |E(f)|^2 over the displayed frequency plane.
    This is the factor that makes steered spots dimmer than the undiffracted one."""
    f = np.fft.fftshift(np.fft.fftfreq(n))
    ex = fill * np.sinc(fill * f) ** 2
    return np.outer(ex, ex)




def spectrum_of(phi):
    """Far-field power of a phase patch, for DISPLAY.
    Rows are flipped so the vertical axis is +f_y (see plot_far_field).
    Returns POWER, already including the element pattern when SPECTRUM_ENVELOPE is on."""
    field = np.exp(1j * phi[::-1, :])
    if SPECTRUM_WINDOW:
        wr = np.hanning(phi.shape[0])[:, None]
        wc = np.hanning(phi.shape[1])[None, :]
        field = field * (wr * wc)
    pw = np.abs(np.fft.fftshift(np.fft.fft2(field)) / phi.size) ** 2
    if SPECTRUM_ENVELOPE and phi.shape[0] == phi.shape[1]:
        pw = pw * element_pattern(phi.shape[0])
    return pw




def pattern_for(theta, period, table, rows, cols, cache):
    """Index map for one steering direction, honouring DERIVE_COMPLEMENTS."""
    if DERIVE_COMPLEMENTS and theta % 360.0 >= 180.0:
        base = (theta - 180.0) % 360.0
        vb = ramp_coordinate(rows, cols, base)
        return np.rot90(build_index_map(vb, period, table,
                                        resolve_shift(base, period, vb, table, cache)), 2)
    v = ramp_coordinate(rows, cols, theta)
    return build_index_map(v, period, table, resolve_shift(theta, period, v, table, cache))




def plot_grid_composite(table, rows, cols, outdir, cache):
    """Incoherent stack of every GRID_SPOTS far field in one image.


    The spots are written as separate frames on the PLM, so their intensities add --
    this is what a time-averaged camera at the Fourier plane would record."""
    if not PLOT_GRID_COMPOSITE:
        return None
    n = GRID_SPECTRUM_N
    r0, c0 = rows // 2 - n // 2, cols // 2 - n // 2
    stack = np.zeros((n, n))
    marks, table_rows = [], []


    for label, theta, period in GRID_SPOTS:
        if period is None:                                   # flat / undiffracted centre
            idx = np.full((rows, cols), FLAT_PHASE_LEVEL, dtype=np.uint8)
            eta_g, env, split, fxa, fya = 1.0, 1.0, 1, 0.0, 0.0
            steer = 0.0
        else:
            idx = pattern_for(theta, period, table, rows, cols, cache)
            v = ramp_coordinate(rows, cols, theta)
            eta_g = grating_efficiency(idx, v, period, table)
            env = pixel_envelope(period, theta)
            split = order_degeneracy(period, theta)
            ct, st = direction_cosines(theta)
            fx, fy = ct / period, st / period
            fxa, fya = fold_to_first_zone(fx), fold_to_first_zone(fy)
            steer = steering_deg(period)


        phi = table.realized(idx[r0:r0 + n, c0:c0 + n])
        stack += spectrum_of(phi)


        eta_tot = eta_g * env / split
        marks.append((label, fxa, fya, eta_tot, split))
        table_rows.append((label, theta, period, eta_g, env, split, eta_tot, steer))


    img = 10 * np.log10(np.maximum(stack / max(stack.max(), 1e-12), 1e-6))
    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=150)
    m = ax.imshow(img, cmap="inferno", vmin=-45, vmax=0, origin="lower",
                  extent=freq_extent(n))
    for label, fx, fy, eta, split in marks:
        ax.plot(fx, fy, "o", mfc="none", mec="cyan", ms=17, mew=1.6)
        dy = 34 if fy < 0 else -34          # keep the label inside the axes
        va = "bottom" if fy < 0 else "top"
        ax.annotate(f"{label}\n{100*eta:.1f} %", xy=(fx, fy), xytext=(0, dy),
                    textcoords="offset points", ha="center", va=va, color="cyan",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="black", ec="cyan", alpha=.7))
    ax.set_xlabel("$f_x$ [cycles/pixel]"); ax.set_ylabel("$f_y$ [cycles/pixel]")
    ax.set_title(f"Steering grid, stacked far field   [{QUANT_RULE} / {LEVEL_PHASE_MODEL}]\n"
                 f"circles = target order, label = $\\eta_{{total}}$",
                 fontweight="bold", fontsize=10)
    fig.colorbar(m, ax=ax, label="dB (normalised to brightest spot)")
    fig.tight_layout()
    path = os.path.join(outdir, "Grid_Composite_FarField.png")
    fig.savefig(path, dpi=200); plt.close(fig)


    print("\nSTEERING GRID")
    print(f"  {'spot':>7} {'theta':>6} {'Lambda':>8} {'eta_grat':>9} {'envel':>7} "
          f"{'split':>6} {'eta_tot':>8} {'steer':>7}")
    for lab, th, pe, g, e, s, t, st_ in table_rows:
        th_s = "flat" if th is None else f"{th:g}"
        pe_s = "-" if pe is None else f"{pe:.4f}"
        print(f"  {lab:>7} {th_s:>6} {pe_s:>8} {100*g:>8.1f}% {100*e:>6.1f}% {s:>6} "
              f"{100*t:>7.1f}% {st_:>6.2f}d")
    steered = [t for lab, th, pe, g, e, s, t, st_ in table_rows if pe is not None]
    print(f"  --> steered spots: mean {100*np.mean(steered):.1f} %   "
          f"min {100*min(steered):.1f} %   max {100*max(steered):.1f} %")
    return path




def plot_far_field(table, rows, cols, outdir, cache):
    if PLOT_FAR_FIELD == "off":
        return []
    if PLOT_FAR_FIELD == "all":
        cases = [(float(a), float(p)) for a in ANGLES_DEG for p in PERIODS]
        print(f"PLOT_FAR_FIELD = 'all' -> rendering {len(cases)} far-field plots, this is slow")
    else:
        cases = SPECTRUM_CASES


    paths, n = [], SPECTRUM_N
    for theta, period in cases:
        v = ramp_coordinate(rows, cols, theta)
        # pattern_for honours DERIVE_COMPLEMENTS, so this plots the map that is actually
        # written for theta >= 180 instead of an independently recomputed one
        idx = pattern_for(theta, period, table, rows, cols, cache)
        r0, c0 = rows // 2 - n // 2, cols // 2 - n // 2
        phi = table.realized(idx[r0:r0 + n, c0:c0 + n])
        # rows run downward in array space but +y points UP, so flip the row axis before
        # the transform. Without this the vertical frequency axis is -f_y and the target
        # marker lands mirrored about f_y = 0 for every non-horizontal grating.
        pw = spectrum_of(phi)
        img = 10 * np.log10(np.maximum(pw / max(pw.max(), 1e-12), 1e-6))


        fig, ax = plt.subplots(figsize=(6, 5.4), dpi=150)
        m = ax.imshow(img, cmap="inferno", vmin=-50, vmax=0, origin="lower",
                      extent=freq_extent(n))


        ct, st = direction_cosines(theta)
        fx, fy = ct / period, st / period
        fxa, fya = fold_to_first_zone(fx), fold_to_first_zone(fy)


        eta_g = grating_efficiency(idx, v, period, table)
        env = pixel_envelope(period, theta)
        split = order_degeneracy(period, theta)
        eta_tot = eta_g * env / split


        ax.plot(fxa, fya, "o", mfc="none", mec="cyan", ms=16, mew=1.8,
                label="+1 order (target)")
        ax.annotate(f"+1\n{100*eta_tot:.1f} %", xy=(fxa, fya),
                    xytext=(10, 10), textcoords="offset points", color="cyan",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="cyan", alpha=.65))


        # sanity annotation: where the brightest non-DC lobe actually sits.
        # img is already materialised and spectrum_of always returns a fresh array,
        # so pw can be zeroed in place.
        f_ax = np.fft.fftshift(np.fft.fftfreq(n))
        pw[n // 2, n // 2] = 0.0                       # ignore DC
        pi_, pj_ = np.unravel_index(int(np.argmax(pw)), pw.shape)
        ax.plot(f_ax[pj_], f_ax[pi_], "+", c="lime", ms=12, mew=1.4, label="measured peak")


        ax.set_xlabel("$f_x$ [cycles/pixel]"); ax.set_ylabel("$f_y$ [cycles/pixel]")
        deg_note = "   [+1/-1 degenerate, power split]" if split == 2 else ""
        ax.set_title(f"Far field  $\\theta$={theta:g}$\\degree$, $\\Lambda$={period:g} px   "
                     f"steer {steering_deg(period):.2f}$\\degree$\n"
                     f"$\\eta_{{grating}}$={100*eta_g:.1f} %  x  envelope={100*env:.1f} %"
                     + ("  /2" if split == 2 else "")
                     + f"  =  $\\eta_{{total}}$={100*eta_tot:.1f} %{deg_note}",
                     fontweight="bold", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(m, ax=ax, label="dB")
        fig.tight_layout()
        p = os.path.join(outdir, "far_field", f"FarField_theta{theta:g}_L{period:g}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200); plt.close(fig)
        paths.append(p)
    return paths




# ================================ HELPERS ================================
def matrix_block(idx_map, size=None):
    size = MATRIX_SIZE if size is None else size
    r, c = idx_map.shape
    if ORIGIN == "center":
        r0, c0 = r // 2 - size // 2, c // 2 - size // 2
    else:
        r0 = c0 = 0
    return idx_map[r0:r0 + size, c0:c0 + size]




def print_block(block, title):
    print(f"--- {title} ---")
    for row in block:
        print(" ".join(f"P{int(k):<2d}" for k in row))
    print("-" * 62)




# ================================== MAIN ==================================
def main():
    rows, cols = PLM_HEIGHT // 2, PLM_WIDTH // 2
    outdir = (f"PLM_Patterns_{PLM_WIDTH}x{PLM_HEIGHT}_{QUANT_RULE}"
              f"_{LEVEL_PHASE_MODEL}_{PHASE_ANCHOR}")
    os.makedirs(outdir, exist_ok=True)


    disp_nm, phase_raw, table = build_phase_levels()
    anchor_offset = float(table.phase[ANCHOR_LEVEL])
    shift_cache = {}          # shared by the symmetry self-test and the main sweep


    print("=" * 78)
    print("PLM BLAZED-GRATING GENERATOR")
    print("=" * 78)
    print(f"bitmap            : {PLM_WIDTH} x {PLM_HEIGHT}   ->  phase array {cols} x {rows}")
    print(f"wavelength        : {LAMBDA_NM:.1f} nm     pixel pitch: {PIXEL_PITCH_UM} um")
    print(f"max piston        : {MAX_DISPLACEMENT_NM:.2f} nm  ->  phase span "
          f"{phase_raw.max()/np.pi:.4f} pi  ({100*phase_raw.max()/TWO_PI:.1f} % of 2 pi)")
    print(f"max quant. error  : {table.max_quantization_error():.4f} rad")
    print(f"phase anchor      : '{PHASE_ANCHOR}'  (ramp origin = {ORIGIN}"
          + (f", pinned to P{ANCHOR_LEVEL} = {anchor_offset/np.pi:.4f} pi)"
             if PHASE_ANCHOR == "P0" else ", offset optimised)"))
    _rule_note = {
        "clip":    "  (truncated blaze: sawtooth resets each period, saturates at P15)",
        "nearest": "  (wrapped blaze: dead-zone phases fold to the closer of P15 / P0)",
        "lut":     "  (THRESHOLDS_PCT used as ramp decision boundaries; anchor ignored)",
    }
    print(f"quantiser rule    : '{QUANT_RULE}'{_rule_note.get(QUANT_RULE, '')}")
    print(f"level phase model : '{LEVEL_PHASE_MODEL}'  ->  full scale = "
          f"{phase_raw.max()/np.pi:.4f} pi")
    if phase_raw.max() < TWO_PI:
        print(f"[NOTE] levels do not reach 2 pi: the blaze cannot wrap perfectly. "
              f"Full 2 pi would need {LAMBDA_NM/2:.1f} nm of piston "
              f"(or lambda <= {2*MAX_DISPLACEMENT_NM:.1f} nm).")
    print()


    # ----------------------- calibration table + CSV -----------------------
    calib_csv = os.path.join(outdir, "Phase_Calibration_P0_P15.csv")
    print("PHASE LEVEL CALIBRATION TABLE (P0 - P15)")
    print("-" * 78)
    print(f"{'Level':<7}{'Code (%)':>12}{'Displ. (nm)':>14}{'Phase (rad)':>13}"
          f"{'Phase (pi)':>13}{'Tile':>10}")
    print("-" * 78)
    with open(calib_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Phase_Tile", "Code_Percent", "Displacement_nm", "Phase_rad",
                    "Phase_pi", "TL", "TR", "BL", "BR"])
        for k in range(16):
            t = BIT_TILES[k]
            print(f"P{k:<6d}{THRESHOLDS_PCT[k]:>12.4f}{disp_nm[k]:>14.3f}"
                  f"{phase_raw[k]:>13.5f}{phase_raw[k]/np.pi:>13.4f}"
                  f"{f'{t[0,0]}{t[0,1]}{t[1,0]}{t[1,1]}':>10}")
            w.writerow([f"P{k}", f"{THRESHOLDS_PCT[k]:.6f}", f"{disp_nm[k]:.4f}",
                        f"{phase_raw[k]:.6f}", f"{phase_raw[k]/np.pi:.6f}",
                        t[0, 0], t[0, 1], t[1, 0], t[1, 1]])
    print("-" * 78)
    print(f"saved: {calib_csv}\n")


    print(f"saved: {plot_calibration(disp_nm, phase_raw, table, outdir)}")
    print(f"saved: {plot_tile_key(phase_raw, outdir)}\n")


    # --------------------------- symmetry self-test ---------------------------
    print("SYMMETRY SELF-TEST (theta+180 sequence == reverse of theta sequence)")
    lines, worst = symmetry_report(table, rows, cols, shift_cache)
    for ln in lines:
        print(ln)
    print(f"  ==> worst case: {worst} mismatches   {'PASS' if worst == 0 else 'FAIL'}\n")


    # ----------------------------- main sweep -----------------------------
    eff_csv = os.path.join(outdir, "Diffraction_Efficiency.csv")
    mat_csv = os.path.join(outdir, f"CGH_Phase_Matrices_{MATRIX_SIZE}x{MATRIX_SIZE}.csv")
    records, n_done = [], 0
    free_shift_cache = {}     # only exercised when the anchor is pinned (eta_free_anchor)
    bmp_dir = os.path.join(outdir, "bmp")
    if SAVE_BMP:
        os.makedirs(bmp_dir, exist_ok=True)
        print(f"SAVE_BMP = True -> writing {len(ANGLES_DEG) * len(PERIODS)} bitmaps "
              f"({PLM_WIDTH}x{PLM_HEIGHT} each) to {bmp_dir}")


    cut_csv = os.path.join(outdir, "Code_Cuts.csv")
    with open(eff_csv, "w", newline="") as f_eff, \
         open(mat_csv, "w", newline="") as f_mat, \
         open(cut_csv, "w", newline="") as f_cut:
        w_eff, w_mat, w_cut = csv.writer(f_eff), csv.writer(f_mat), csv.writer(f_cut)
        w_eff.writerow(["theta_deg", "period_px", "anchor", "quant_rule", "source", "shift_frac",
                        "levels_used",
                        "eta_grating", "eta_free_anchor", "eta_legacy", "eta_envelope",
                        "order_split", "eta_total", "steer_deg", "nyquist_ok",
                        "alias_period_px", "alias_reversed"])
        w_mat.writerow(["type", "theta_deg", "period_px", "codes_used", "eta_grating", "row"]
                       + [f"c{c}" for c in range(MATRIX_SIZE)])
        w_cut.writerow(["theta_deg", "period_px", "cut", "source"]
                       + [f"s{c}" for c in range(CUT_LEN)])


        if DERIVE_COMPLEMENTS:
            base_angles = [a for a in ANGLES_DEG if a % 360.0 < 180.0]
            complement_set = {a % 360.0 for a in ANGLES_DEG if a % 360.0 >= 180.0}
            print(f"DERIVE_COMPLEMENTS: computing {base_angles} and deriving "
                  f"{sorted(complement_set)} as rot180 of them\n")
        else:
            base_angles, complement_set = list(ANGLES_DEG), set()


        def emit(theta, period, idx, v, v_lab, shift, derived):
            """Score, log and (optionally) save one pattern."""
            nonlocal n_done
            eta_g = grating_efficiency(idx, v, period, table)
            if PHASE_ANCHOR == "free" or QUANT_RULE != "nearest":
                eta_f = eta_g
            else:
                key = (round(theta % 180.0, 6), period)
                if key not in free_shift_cache:
                    free_shift_cache[key] = optimise_shift(v, period, table)
                eta_f = grating_efficiency(
                    build_index_map(v, period, table, free_shift_cache[key]),
                    v, period, table)
            eta_l = grating_efficiency(make_index_map_legacy(v_lab, period), v_lab, period, table)
            env = pixel_envelope(period, theta)
            split = order_degeneracy(period, theta)
            eta_order = eta_g / split              # power actually reaching ONE first order
            ok, alias_p, rev = alias_info(period, theta)
            used = "|".join(f"P{k}" for k in np.unique(idx))
            src = "rot180" if derived else "computed"


            records.append({"theta": theta, "period": period, "eta_grating": eta_g,
                            "eta_total": eta_order * env, "eta_free": eta_f,
                            "eta_legacy": eta_l, "split": split})
            w_eff.writerow([f"{theta:g}", f"{period:.2f}", PHASE_ANCHOR, QUANT_RULE, src,
                            f"{shift:.6f}", used,
                            f"{eta_g:.6f}", f"{eta_f:.6f}", f"{eta_l:.6f}", f"{env:.6f}",
                            split, f"{eta_order*env:.6f}", f"{steering_deg(period):.4f}",
                            int(ok), f"{alias_p:.4f}", int(rev)])


            if SAVE_BMP:
                Image.fromarray(index_map_to_bitmap(idx) * 255).save(
                    os.path.join(bmp_dir, f"CGH_PLM_L{period:.1f}_theta{theta:g}deg.bmp"))
            n_done += 1


            blk = matrix_block(idx)
            if PRINT_MATRIX and round(period, 2) in LOG_LAMBDAS:
                note = "  [SYMMETRIC: +1/-1 split, cannot steer one way]" if split == 2 else ""
                print_block(blk, f"theta={theta:g} deg, Lambda={period:.1f} px  |  "
                                 f"codes {used}  |  eta={100*eta_g:5.1f} % "
                                 f"(original {100*eta_l:5.1f} %)  shift={shift:.4f} [{src}]{note}")
            if CSV_MATRIX:
                for r_i, row in enumerate(blk):
                    w_mat.writerow(["grating", f"{theta:g}", f"{period:.2f}", used,
                                    f"{eta_g:.6f}", r_i] + [f"P{int(k)}" for k in row])
            if CSV_CUTS:
                half = CUT_LEN // 2
                # sample the same part of the panel that matrix_block logs
                ri, ci = (rows // 2, cols // 2) if ORIGIN == "center" else (half, half)
                h = idx[ri, ci - half:ci + half + 1]
                w = idx[ri - half:ri + half + 1, ci]
                w_cut.writerow([f"{theta:g}", f"{period:.2f}", "row_horizontal", src]
                               + [f"P{int(k)}" for k in h])
                w_cut.writerow([f"{theta:g}", f"{period:.2f}", "col_vertical", src]
                               + [f"P{int(k)}" for k in w])


        for theta in base_angles:
            comp = (theta + 180.0) % 360.0
            has_comp = DERIVE_COMPLEMENTS and comp in complement_set


            if theta in FLAT_ANGLES:
                flat = np.full((rows, cols), FLAT_PHASE_LEVEL, dtype=np.uint8)
                if SAVE_BMP:
                    Image.fromarray(index_map_to_bitmap(flat) * 255).save(
                        os.path.join(bmp_dir, f"CGH_PLM_Flat_P{FLAT_PHASE_LEVEL}_"
                                              f"{PLM_WIDTH}x{PLM_HEIGHT}.bmp"))
                blk = matrix_block(flat)
                # a flat panel is its own rot180, so the complement is emitted here too --
                # otherwise DERIVE_COMPLEMENTS drops theta+180 from the run entirely
                for th in [theta] + ([comp] if has_comp else []):
                    n_done += 1
                    if PRINT_MATRIX:
                        print_block(blk, f"flat state P{FLAT_PHASE_LEVEL}  (theta={th:g} deg)")
                    if CSV_MATRIX:
                        for r_i, row in enumerate(blk):
                            w_mat.writerow(["flat", f"{th:g}", "NA", f"P{FLAT_PHASE_LEVEL}",
                                            "NA", r_i] + [f"P{int(k)}" for k in row])
                continue


            v = ramp_coordinate(rows, cols, theta)
            v_lab = ramp_coordinate(rows, cols, theta, origin="corner")  # original-script convention
            # direction_cosines negates exactly for theta+180, so the complementary ramps
            # are exactly -v and do not need rebuilding once per period
            v_comp     = -v     if has_comp else None
            v_comp_lab = -v_lab if has_comp else None


            for period in PERIODS:
                period = float(period)
                shift = resolve_shift(theta, period, v, table, shift_cache)
                idx = build_index_map(v, period, table, shift)
                emit(theta, period, idx, v, v_lab, shift, derived=False)


                if has_comp:
                    emit(comp, period, np.rot90(idx, 2), v_comp, v_comp_lab,
                         shift, derived=True)
                del idx


            for th in [theta] + ([comp] if has_comp else []):
                d = [r for r in records if r["theta"] == th]
                if d:
                    print(f"theta = {th:6.1f} deg   mean eta = "
                          f"{100*np.mean([r['eta_grating'] for r in d]):5.1f} %   "
                          f"(original {100*np.mean([r['eta_legacy'] for r in d]):5.1f} %)"
                          + ("   [derived by rot180]" if th != theta else ""))
    print(f"\nsaved: {eff_csv}")
    print(f"saved: {mat_csv}")
    print(f"saved: {cut_csv}")


    deg = sorted({(r["theta"], r["period"]) for r in records if r["split"] == 2})
    if deg:
        pers = sorted({p for _, p in deg})
        print(f"\n[SYMMETRIC GRATINGS] {len(deg)} case(s) at Lambda = "
              f"{', '.join(f'{p:g}' for p in pers)} px sit exactly at Nyquist.")
        print("  The +1 and -1 orders coincide on the sampled grid, so the pattern is its own")
        print("  mirror image: theta and theta+180 give IDENTICAL codes and the power splits")
        print("  50/50 between the two first orders. eta_total is halved accordingly.")
        print("  Use Lambda > 2 px if you need the beam to go one way only.")


    # ------------------------- complementary check -------------------------
    print("\nCOMPLEMENTARY-ANGLE EFFICIENCY CHECK  |eta(theta) - eta(theta+180)|")
    by = {(r["theta"], r["period"]): r["eta_grating"] for r in records}
    for theta in [0.0, 45.0, 90.0, 135.0]:
        d = [abs(by[(theta, p)] - by[(theta + 180.0, p)]) for p in [float(x) for x in PERIODS]
             if (theta, p) in by and (theta + 180.0, p) in by]
        if d:
            print(f"  {theta:6.1f} vs {theta+180:6.1f} deg : max diff = {max(d):.3e}")


    # ------------------------------- plots -------------------------------
    for p in plot_efficiency(records, outdir):
        print(f"saved: {p}")
    gc = plot_grid_composite(table, rows, cols, outdir, shift_cache)
    if gc:
        print(f"saved: {gc}")


    ff = plot_far_field(table, rows, cols, outdir, shift_cache)
    if ff:
        print(f"saved: {len(ff)} far-field plot(s) -> {os.path.dirname(ff[0])}")


    v0 = ramp_coordinate(rows, cols, 0.0)
    rp = plot_phase_ramps(table, outdir,
                          lambda p: resolve_shift(0.0, p, v0, table, shift_cache))
    if rp:
        print(f"saved: {len(rp)} phase-ramp plot(s), overview -> {rp[0]}")


    eg = np.array([r["eta_grating"] for r in records])
    ef = np.array([r["eta_free"] for r in records])
    el = np.array([r["eta_legacy"] for r in records])
    ok = np.array([r["period"] >= 2.0 for r in records])
    print(f"\n{n_done} patterns processed (BMP saving = {SAVE_BMP}).")
    print(f"mean grating efficiency, Lambda >= 2 px : {100*eg[ok].mean():.2f} %  "
          f"(original {100*el[ok].mean():.2f} %)")
    if PHASE_ANCHOR != "free":
        print(f"cost of pinning the ramp to P{ANCHOR_LEVEL}          : "
              f"{100*(ef[ok].mean() - eg[ok].mean()):.2f} percentage points")
    print(f"best : {100*eg.max():.2f} %    worst (Lambda >= 2): {100*eg[ok].min():.2f} %")




if __name__ == "__main__":
    main()
