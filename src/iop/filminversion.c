/*
    This file is part of darktable,
    Copyright (C) 2020-2024 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "bauhaus/bauhaus.h"
#include "common/colorlabels.h"
#include "common/darktable.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/openmp_maths.h"
#include "develop/pixelpipe_hb.h"
#include "dtgtk/button.h"
#include "dtgtk/paint.h"
#include "dtgtk/resetlabel.h"
#include "dtgtk/togglebutton.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "gui/color_picker_proxy.h"
#include "iop/iop_api.h"

#include <glib.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Converts a linear scene-referred film-negative (or slide) scan into a
 * linear scene-referred positive via a density-space H&D characteristic
 * curve, plus mask-neutralization and grading controls.
 *
 * Sits after demosaic/white-balance, before exposure/filmic.
 *   INPUT:  linear scene-referred RGB, not inverted, not clipped to [0,1].
 *   DOMAIN: density = -log10(clamp(linear, eps)); all shaping happens here.
 *   OUTPUT: linear = 10^-density, rescaled so the curve pivot lands on 18%
 *           mid-gray. Deliberately NOT clipped -- downstream filmic/tone
 *           mapping owns highlight handling.
 *
 * The curve is an asymmetric toe-linear-shoulder response: a straight line
 * of slope k through the pivot, bounded by two independent softplus knees
 * (shoulder -> paper white, toe -> paper black), giving a monotonic curve
 * differentiable everywhere:
 *   v1      = D_min_eff + softplus(a_hl * (v - D_min_eff)) / a_hl
 *   D_print = D_max_eff - softplus(a_sh * (D_max_eff - v1)) / a_sh
 *
 * The curve math and most constants below are adapted from NegPy
 * (https://github.com/marcinz606/NegPy), Copyright the NegPy authors,
 * licensed GPLv3, reused here under GPLv3.
 **/

#define DENSITY_EPS 1e-10f
#define HD_D_MIN 0.06f                 // paper white floor (NegPy default)
#define HD_D_MAX 2.3f                  // physical paper black ceiling (NegPy default)
#define HD_TOE_SHARPNESS_BASE 4.0f     // NegPy toe_sharpness_base
#define HD_SHOULDER_SHARPNESS_BASE 3.0f// NegPy shoulder_sharpness_base
#define HD_TOESHOULDER_WIDTH_REF 2.5f  // NegPy toeshoulder_width_ref
#define HD_TOE_HEIGHT 0.90f            // NegPy toe_height
#define HD_SHOULDER_HEIGHT 0.35f       // NegPy shoulder_height
#define HD_GRADE_CONTRAST_SCALE 2.2f   // NegPy grade_contrast_scale

// Grade-coupled toe/shoulder: hard grades (high slope) physically have
// snappier toes and more compressed shoulders. slope_norm runs 0 (softest
// grade) to 1 (hardest); applied unconditionally, matching NegPy
// (grade_coupled_shape has no on/off toggle either).
#define HD_TOE_GRADE_STRENGTH (0.15f * 0.35f / 0.90f) // NegPy toe_grade_strength
#define HD_SHOULDER_GRADE_STRENGTH 0.12f               // NegPy shoulder_grade_strength

// Print density the pivot is solved onto, independent of grade/toe/shoulder
// (NegPy anchor_target_density). A fixed density deep in the straight-line
// region keeps density/exposure predictable instead of drifting into the
// shoulder knee. See _solve_v_star().
#define HD_ANCHOR_TARGET_DENSITY 0.75f
// Zone Density / Split Grade: mid-sparing sigmoid weights centred in the
// three-quarter/quarter tones, so midtones get neither effect.
#define HD_ZONE_DENSITY_SHARPNESS 4.0f
#define HD_ZONE_SHADOW_CENTER 1.5f     // HD_ANCHOR_TARGET_DENSITY + 0.75
#define HD_ZONE_HIGHLIGHT_CENTER 0.35f // HD_ANCHOR_TARGET_DENSITY - 0.40
// Snap (NegPy midtone_gamma): paper baseline gamma boost at the midtone
// centre; the "snap" slider is an additive trim on top.
#define HD_PAPER_MIDTONE_GAMMA 0.0f
#define HD_PAPER_GAMMA_WIDTH 0.6f

// Dye Separation Damping: print-density chroma at which the per-pixel gain
// crosses 1.0; below it muted colour gets the full separation push, above it
// vivid colour is pulled back. Mirrors NegPy separation_damping_ref_spread.
#define HD_SEPARATION_DAMPING_REF_SPREAD 0.35f

// Section 3B (automatic per-channel floor/ceiling normalization): reduces
// the frame to a coarse block-median grid, then measures three things from
// it -- the wide floor/ceiling normalization range, its luma-weighted
// spread (floor_ceil_range), and the tighter P10-P90 "textural" range. The
// latter two feed HD_AUTO_GRADE_*'s damping formula, which is what sets the
// slope-scaling density_range, NOT either raw measurement directly.
#define HD_GRID_TARGET 1024        // NegPy analysis_grid
// Fraction of width/height trimmed off each edge before building the grid,
// so uncropped scan borders (rebate, sprocket holes) don't pollute the
// statistics. Only a cheap first pass; anything larger than this thin strip
// is caught by the surround rejection below.
#define HD_ANALYSIS_EDGE_MARGIN 0.03f
// Scans routinely include far more than that: the film rebate, the holder,
// or plain negative space around the frame. Such regions are optically flat
// and sit at one extreme of the density range, so they drag the floor and
// ceiling percentiles -- and with them the neutral axis -- away from the
// picture. A grid cell is discarded when it carries no texture, is reachable
// from the frame border, and sits clearly away from the textured content.
#define HD_REBATE_FLATNESS 0.010f   // mean |neighbour delta| in log10 density
#define HD_REBATE_SEPARATION 0.15f  // log10 density away from the content median
#define HD_REBATE_MIN_KEEP 0.20f    // never discard more than 80% of the grid
#define HD_REBATE_MIN_CELLS 64      // nor leave fewer cells than this to measure
// NegPy's own base_luma_clip=0.01 is a *percentile* value (0.01st/99.99th),
// not a percent -- two orders of magnitude less aggressive than this. 1st/99th
// is deliberately tighter here: darktable exposes no equivalent of NegPy's
// manual analysis_rect crop, so more aggressive tail-clipping is the safer
// default against dust/hot pixels/uncropped scan borders.
#define HD_FLOOR_PERCENTILE 1.0f
#define HD_CEIL_PERCENTILE 99.0f
#define HD_TEXTURAL_LO_PERCENTILE 10.0f  // NegPy textural_range_clip=10.0
#define HD_TEXTURAL_HI_PERCENTILE 90.0f
#define HD_TEXTURAL_RANGE_MIN 0.3f
#define HD_TEXTURAL_RANGE_MAX 3.5f
// Deliberately lower than NegPy's identical slope_min=2.0. At default
// settings this module's base_slope (HD_GRADE_CONTRAST_SCALE * density_range)
// sits right around 2.0 for a nominal image -- a side effect of
// HD_GRADE_CONTRAST_SCALE having been lowered from NegPy's 2.9 for a softer
// baseline look, which left grade/contrast-trim almost no headroom to go
// softer before clamping here. Lowering the floor restores that headroom
// without reverting the softer-baseline choice; still prevents genuinely
// degenerate near-flat curves.
#define HD_SLOPE_MIN 1.0f          // NegPy slope_min = 2.0
#define HD_SLOPE_MAX 10.0f         // NegPy slope_max

// Holds printed midtone contrast partially constant by damping the
// floor_ceil/textural ratio toward a canonical "normal negative" ratio, so
// an unusually flat or contrasty frame doesn't swing the slope as hard as
// its raw measured range would.
#define HD_AUTO_GRADE_TARGET 0.6f
#define HD_AUTO_GRADE_STRENGTH 0.5f
#define HD_AUTO_GRADE_NOMINAL_RATIO 2.0f

// The reference tone (a typical negative's normalized median) the pivot is
// centred on, and how strongly the density slider shifts it.
#define HD_ASSUMED_ANCHOR 0.46f
#define HD_DENSITY_MULTIPLIER 0.2f

// Auto exposure anchor: HD_ASSUMED_ANCHOR is a generic guess, so a fixed
// anchor alone leaves a systematic bias on scans whose midtone differs.
// Blend in a clamped fraction of the frame's own median luma to correct it
// without letting one unusual frame swing exposure too far.
#define HD_ANCHOR_METER_PERCENTILE 50.0f
#define HD_ANCHOR_METER_STRENGTH 0.2f
#define HD_ANCHOR_METER_BAND 0.12f

// Filtration is scaled by this and divided by the measured per-channel
// range, so a slider unit stays consistent across scans.
#define HD_CMY_MAX_DENSITY 0.2f

// Regional CMY (NegPy shadow_cyan/magenta/yellow, highlight_*): per-channel
// creative color tint localized to the shadow/highlight zones, distinct from
// Split Grade (achromatic contrast) and Zone Density (achromatic brightness).
// Applied directly to v (same domain as those two, and as Snap), via a
// sigmoid centered on HD_ANCHOR_TARGET_DENSITY -- one shared center, unlike
// Split Grade/Zone Density's two independent zone centers, so w_sh + w_hi
// always sum to 1. NegPy hardcodes this sharpness independently of
// HD_ZONE_DENSITY_SHARPNESS.
#define HD_CMY_ZONE_SHARPNESS 3.0f

// Temperature: a Kelvin lever over filtration's green/blue (magenta/yellow)
// pair (NegPy wb_to_kelvin/kelvin_to_wb) -- not a stored param of its own,
// just an alternate, more intuitive way to dial in the same two sliders.
// Projects the current (magenta, yellow) onto the Planckian (mired)
// direction and slides along it; the perpendicular off-locus tint is left
// untouched, and red/cyan is never touched (real darkroom: cyan stays 0).
#define HD_TEMP_REF_KELVIN 5500.0f
#define HD_TEMP_MIN_KELVIN 3000.0f
#define HD_TEMP_MAX_KELVIN 12000.0f
// Slider units per mired, red-anchored: Wien slopes at ~460/550/690nm over a
// nominal print gamma (~4) and HD_CMY_MAX_DENSITY -- so the Kelvin readout
// is nominal, not colorimetric. Ported as-is from NegPy: both use the same
// -1..1 slider domain scaled by the same 0.2 cmy_max_density constant.
#define HD_TEMP_K_MAGENTA 0.0029f
#define HD_TEMP_K_YELLOW 0.0057f

// toe/shoulder are scaled by this before use, for both the d_max_eff/
// d_min_eff height lift and the negative-branch knee sharpening.
#define HD_TOE_SHOULDER_STRENGTH 0.85f

// Cast Removal (section 3C): per-channel slope/pivot/curvature fit so red
// and blue match green's curve at near-neutral reference points in the
// shadow/mid/highlight bands, C41 only.
//
// Band-direction note: NegPy's shadow/highlight band constants use its
// internal normalized convention, which is the OPPOSITE of ours (here:
// floor/low-density = shadow). The ranges below are NegPy's numbers with
// the shadow/highlight LABELS swapped so they land on the correct tones.
#define HD_CAST_SHADOW_BAND_LO 0.10f
#define HD_CAST_SHADOW_BAND_HI 0.30f
#define HD_CAST_MID_BAND_LO 0.40f
#define HD_CAST_MID_BAND_HI 0.60f
#define HD_CAST_HIGHLIGHT_BAND_LO 0.72f
#define HD_CAST_HIGHLIGHT_BAND_HI 0.92f
#define HD_CAST_CHROMA_QUANTILE 0.30f   // lowest-chroma fraction of each band kept
#define HD_CAST_FIRST_PASS_CAP 0.55f    // pass-1 (pre-correction) chroma ceiling
#define HD_CAST_CHROMA_CAP 0.29f        // pass-2 (corrected) chroma ceiling
#define HD_CAST_MIN_CELLS 64            // min grid cells for a band to be usable
#define HD_CAST_CONFIDENCE_N0 256.0f    // confidence sample-size half-point
#define HD_CAST_AGREEMENT_DEADZONE 0.10f
#define HD_CAST_AGREEMENT_SCALE 0.20f
#define HD_CAST_MIDTONE_MAX_OFFSET 0.2f // clamp on each channel's ref deviation from green
#define HD_CAST_CURV_MAX_RATIO 0.45f    // curvature clamp, fraction of slope
#define HD_CAST_SHADOW_TIE_PERCENTILE 2.0f  // 1-point fallback ref, on our floor side
#define HD_CAST_MAX_OFFSET 0.1f         // 1-point fallback tilt clamp


DT_MODULE_INTROSPECTION(8, dt_iop_filminversion_params_t)


typedef enum dt_iop_filminversion_mode_t
{
  DT_FILMINVERSION_C41 = 0, // $DESCRIPTION: "color negative (C-41)"
  DT_FILMINVERSION_BW = 1,  // $DESCRIPTION: "black & white negative"
  DT_FILMINVERSION_E6 = 2   // $DESCRIPTION: "slide / reversal (E-6)"
} dt_iop_filminversion_mode_t;


typedef struct dt_iop_filminversion_params_t
{
  dt_iop_filminversion_mode_t mode; /* $DEFAULT: DT_FILMINVERSION_C41 $DESCRIPTION: "film type" */

  /* mask neutralization: per-channel density offset (e.g. orange base of a
     color negative), subtracted in density space before the curve. */
  float mask_offset[4];  /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "mask offset" */

  /* creative filtration: per-channel additive density trim applied after
     mask neutralization (mimics enlarger dichroic-head CMY filtration,
     which is itself an additive density-unit shift). */
  float filtration[4];   /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "filtration" */

  /* per-channel slope trim: dye layers don't share a gamma, so a shared
     curve plus a constant offset can only neutralize one tonal point --
     elsewhere a density-dependent cast remains ("crossover"). */
  float k_trim[4];        /* $MIN: 0.5 $MAX: 2.0 $DEFAULT: 1.0 $DESCRIPTION: "contrast trim" */

  /* see _measure_neutral_axis()/_solve_cast_removal(). Matches NegPy's own
     Cast Removal slider exactly: 0..1, default 0.5 -- the raw value feeds
     the fit directly, no remapping, same as NegPy's cast_removal_strength. */
  float cast_removal_strength; /* $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.5 $DESCRIPTION: "cast removal" */

  /* global density/exposure shift applied before the curve is evaluated */
  float density;         /* $MIN: -0.5 $MAX: 1.5 $DEFAULT: 0.5 $DESCRIPTION: "density" */

  /* Zone Density: brightness offset localized to the shadow/highlight zones
     only. Ranges are asymmetric because density is log10 -- an equal |delta|
     reads far smaller near d_max than near d_min. */
  float shadow_density;     /* $MIN: -0.9 $MAX: 0.9 $DEFAULT: 0.0 $DESCRIPTION: "shadows density" */
  float highlight_density;  /* $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "highlights density" */

  /* H&D curve parameters (see documentation block above) */
  float grade;            /* $MIN: 105.0 $MAX: 150.0 $DEFAULT: 127.5 $DESCRIPTION: "grade (ISO-R)" */

  /* Split Grade: local contrast trim in grade's ISO-R units, applied only
     in the shadow/highlight zones. */
  float shadow_grade;      /* $MIN: -50.0 $MAX: 50.0 $DEFAULT: 0.0 $DESCRIPTION: "shadows grade" */
  float highlight_grade;   /* $MIN: -50.0 $MAX: 50.0 $DEFAULT: 0.0 $DESCRIPTION: "highlights grade" */

  /* Snap: additive trim on the paper's variable-gamma S-curve at the
     midtone -- positive snaps contrast in, negative flattens. */
  float snap;              /* $MIN: -0.35 $MAX: 0.65 $DEFAULT: 0.0 $DESCRIPTION: "snap" */

  float toe;               /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "toe" */
  float shoulder;          /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "shoulder" */
  float toe_width;         /* $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "toe width" */
  float shoulder_width;    /* $MIN: 0.1 $MAX: 5.0 $DEFAULT: 1.0 $DESCRIPTION: "shoulder width" */

  /* anchor trim applied after the curve */
  float exposure;         /* $MIN: 0.125 $MAX: 8.0 $DEFAULT: 1.0 $DESCRIPTION: "anchor trim" */

  // Dye Separation: density-domain saturation applied to the print curve
  // output. 0 = identity, 1 = full separation (k = 2.0); the matrix is
  // generated from k as k*I + (1-k)*J (rows sum to 1, preserving neutrals).
  float dye_separation;    /* $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "dye separation" */
  float separation_damping; /* $MIN: 0.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "separation damping" */

  /* Regional CMY: per-channel creative color tint localized to the shadow/
     highlight zones (see HD_CMY_ZONE_SHARPNESS), distinct from Split Grade
     and Zone Density above, which are both achromatic. */
  float shadow_filtration[4];    /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "shadows filtration" */
  float highlight_filtration[4]; /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "highlights filtration" */

  /* Per-channel trims on top of the global toe/shoulder/snap/dye separation
     controls above: dye layers don't share a gamma, so a shared curve alone
     can leave a density-dependent crossover cast at the shadow/highlight
     endpoints or midtone that mask offset/filtration/contrast trim cannot
     fix (those only correct a single tonal point or the whole curve
     uniformly). Added to the global value, then clamped to the same domain
     the global slider itself is clamped to -- see _compute_curve_shape(). */
  float toe_trim[4];            /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "toe trim" */
  float shoulder_trim[4];       /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "shoulder trim" */
  float toe_width_trim[4];      /* $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "toe width trim" */
  float shoulder_width_trim[4]; /* $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.0 $DESCRIPTION: "shoulder width trim" */
  float snap_trim[4];           /* $MIN: -0.5 $MAX: 0.5 $DEFAULT: 0.0 $DESCRIPTION: "snap trim" */
  float dye_separation_trim[4]; /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "dye separation trim" */
} dt_iop_filminversion_params_t;


typedef struct dt_iop_filminversion_data_t
{
  dt_iop_filminversion_mode_t mode;
  dt_aligned_pixel_t mask_offset;
  dt_aligned_pixel_t filtration;
  dt_aligned_pixel_t k_trim;
  dt_aligned_pixel_t shadow_filtration;    // HD_CMY_MAX_DENSITY-scaled
  dt_aligned_pixel_t highlight_filtration; // HD_CMY_MAX_DENSITY-scaled
  float cast_removal_strength;
  float density;
  float exposure;
  float shadow_density;
  float highlight_density;

  /* precomputed curve constants (depend only on grade/toe/shoulder/widths,
     not on pixel data -- computed once in commit_params) */
  float grade_slope_factor; // HD_GRADE_CONTRAST_SCALE / (grade/100); the slope
                             // also needs the per-frame density_range, mixed in
                             // once per _compute_frame() call.
  // Toe/shoulder shape (a_sh/a_hl/d_min_eff/d_max_eff/v_star/anchor_scale) is
  // NOT precomputed here: grade-coupled toe/shoulder needs the per-frame
  // measured slope, so only the raw slider inputs are stashed, and the solve
  // itself moves to _compute_curve_shape(), called from
  // _frame_from_measurement(). See dt_iop_filminversion_frame_t.
  float toe;
  float shoulder;
  float toe_width;       // pre-clamped to >= 0.1
  float shoulder_width;  // pre-clamped to >= 0.1
  // Per-channel trims on top of the four raw values above (NegPy
  // per_channel_toe_shoulder/per_channel_widths): combined and clamped in
  // _compute_curve_shape(), not here, since the height trims combine with
  // the grade-coupled effective toe/shoulder, which isn't known yet either.
  dt_aligned_pixel_t toe_trim;
  dt_aligned_pixel_t shoulder_trim;
  dt_aligned_pixel_t toe_width_trim;
  dt_aligned_pixel_t shoulder_width_trim;
  float shadow_grade_gain;    // Split Grade shadow local-contrast gain
  float highlight_grade_gain; // Split Grade highlight local-contrast gain
  float midtone_gamma;   // Snap: HD_PAPER_MIDTONE_GAMMA + p->snap
  dt_aligned_pixel_t snap_trim; // per-channel additive trim on top of the above

  dt_aligned_pixel_t dye_k;    // per-channel 1.0 + clamp(dye_separation + trim, 0, 1)
  float separation_damping;
} dt_iop_filminversion_data_t;

// Per-frame quantities from the whole-frame measurement, recomputed once
// per _compute_frame() call. Separate from dt_iop_filminversion_data_t
// because these depend on the pixel buffer, not just the params.
typedef struct dt_iop_filminversion_frame_t
{
  dt_aligned_pixel_t floor;          // per-channel measured density floor
  dt_aligned_pixel_t ceil;           // per-channel measured density ceiling
  dt_aligned_pixel_t slope;          // per-channel clamped slope (color modes)
  dt_aligned_pixel_t pivot;          // per-channel pivot (NegPy compute_pivot)
  dt_aligned_pixel_t curv;           // per-channel Cast Removal curvature (0 unless active)
  dt_aligned_pixel_t filtration_eff; // per-channel HD_CMY_MAX_DENSITY-scaled filtration
  float floor_bw, ceil_bw;           // luma-combined equivalents for B&W mode
  float slope_bw, pivot_bw;

  // Per-frame curve shape (see _compute_curve_shape()): grade-coupled AND
  // per-channel-trimmed toe/shoulder makes these vary per channel, used by
  // the C41/E6 path. B&W (no per-channel trims) uses the _bw scalar versions
  // instead, which are also what v_star/anchor_scale are solved from.
  dt_aligned_pixel_t a_sh;       // toe knee sharpness
  dt_aligned_pixel_t a_hl;       // shoulder knee sharpness
  dt_aligned_pixel_t d_min_eff;  // shoulder-adjusted paper white
  dt_aligned_pixel_t d_max_eff;  // toe-adjusted paper black
  dt_aligned_pixel_t bpc_black;  // black point compensation clip
  float a_sh_bw, a_hl_bw, d_min_eff_bw, d_max_eff_bw, bpc_black_bw;
  float v_star;          // pivot value solved to hit HD_ANCHOR_TARGET_DENSITY
  float anchor_scale;    // normalizes the curve pivot to 18% mid-gray
} dt_iop_filminversion_frame_t;

// Cast Removal neutral-axis measurement result, see _measure_neutral_axis().
typedef struct dt_iop_filminversion_neutral_axis_t
{
  gboolean valid;                    // mid+shadow bands both found
  gboolean has_highlight;            // highlight band also found (enables curvature)
  float confidence;                  // 0..1, scales cast_removal_strength
  dt_aligned_pixel_t mid_ref;        // per-channel raw log density at the midtone ref
  dt_aligned_pixel_t shadow_ref;     // per-channel raw log density at the shadow ref
  dt_aligned_pixel_t highlight_ref;  // per-channel raw log density at the highlight ref
} dt_iop_filminversion_neutral_axis_t;

// Everything _measure() derives from the input buffer. Depends only on the
// pixels, never on this module's params, which is what makes it cacheable
// across calls and shareable between pipes.
typedef struct dt_iop_filminversion_measurement_t
{
  dt_aligned_pixel_t floor;       // per-channel measured density floor
  dt_aligned_pixel_t ceil;        // per-channel measured density ceiling
  dt_aligned_pixel_t shadow_tie;  // Cast Removal one-point fallback reference
  float density_range;            // damped spread driving the slope scaling
  float anchor;                   // auto exposure anchor
  dt_iop_filminversion_neutral_axis_t na;
} dt_iop_filminversion_measurement_t;


typedef struct dt_iop_filminversion_gui_data_t
{
  GtkNotebook *notebook;
  GtkWidget *mode;
  GtkWidget *mask_offset_R, *mask_offset_G, *mask_offset_B;
  GtkWidget *auto_mask_button;
  GtkWidget *filtration_R, *filtration_G, *filtration_B;
  // Temperature: Kelvin lever over filtration/shadow_filtration/
  // highlight_filtration's green+blue (magenta/yellow) pair, no param of
  // its own. See _wb_to_kelvin()/_kelvin_to_wb()/_update_temperature_sliders().
  GtkWidget *temperature, *shadow_temperature, *highlight_temperature;
  GtkWidget *cast_removal_strength;
  GtkWidget *density;
  GtkWidget *shadow_density, *highlight_density;
  // Regional color: same R/G/B channel-select-tabs idiom as the
  // per-channel trims panel below (see trim_channel_box et al.), just for
  // shadow_filtration/highlight_filtration instead.
  GtkWidget *filtration_channel_box;
  GtkWidget *filtration_channel_R, *filtration_channel_G, *filtration_channel_B;
  int filtration_channel; // 0=R, 1=G, 2=B
  GtkWidget *shadow_filtration, *highlight_filtration;
  GtkWidget *grade;
  GtkWidget *shadow_grade, *highlight_grade;
  GtkWidget *snap;
  GtkWidget *toe, *shoulder, *toe_width, *shoulder_width;
  GtkWidget *exposure;
  GtkWidget *dye_separation;
  GtkWidget *separation_damping;
  // Per-channel trims on top of the six controls above: one round R/G/B
  // channel-select button row (dt_trim_channel below tracks which is active)
  // plus a single shared slider per control, whose displayed value/target
  // swaps to that channel -- replaces having 3x as many always-visible
  // sliders. See _update_trim_sliders()/_trim_channel_callback().
  GtkWidget *trim_channel_box;
  GtkWidget *trim_channel_R, *trim_channel_G, *trim_channel_B;
  int trim_channel; // 0=R, 1=G, 2=B
  GtkWidget *k_trim, *toe_trim, *shoulder_trim, *toe_width_trim, *shoulder_width_trim, *snap_trim;
  GtkWidget *dye_separation_trim_R, *dye_separation_trim_G, *dye_separation_trim_B;
  // latest whole-frame floor/ceiling, so apply_auto_mask_neutralize() can
  // convert a picked patch into the domain the curve uses.
  dt_aligned_pixel_t last_floor, last_ceil;

  // Whole-frame measurement cache: written by the preview pipe, read by
  // every GUI-attached pipe. Keyed on the upstream pipe hash plus the
  // measured region, since a downstream crop changes the region we are
  // handed without ever touching the upstream hash.
  gboolean measure_cache_valid;
  dt_hash_t measure_hash;
  int measure_width, measure_height;
  dt_imgid_t measure_imgid;
  dt_iop_filminversion_measurement_t measurement;
} dt_iop_filminversion_gui_data_t;

typedef struct dt_iop_filminversion_global_data_t
{
  int kernel_filminversion;
} dt_iop_filminversion_global_data_t;


const char *name()
{
  return _("film inversion");
}

const char *aliases()
{
  return _("film|invert|negative|scan|scene-referred");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("invert film negative scans to linear scene-referred RGB"),
                                      _("corrective"),
                                      _("linear, RGB, scene-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, scene-referred"));
}

// No tiling: _compute_frame() measures whole-frame statistics, so per-tile
// buffers would give each tile its own floor/ceiling and produce seams.
int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_ONE_INSTANCE;
}


int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL;
}


dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  // Pre-5 (originally just version 2): the module's very first prototype --
  // an unrelated power-curve algorithm (Dmin/wb_high/wb_low/D_max/offset/
  // contrast), replaced wholesale by the current H&D-curve/NegPy-ported one
  // before any per-field migration was ever written. No principled mapping
  // exists between the two, so just fall back to today's defaults rather
  // than leaving dt_dev_read_history_ext() to discard the whole history
  // entry with a "version mismatch" log (see develop.c's dev_read_history).
  if(old_version < 5)
  {
    dt_iop_filminversion_params_t *n = (dt_iop_filminversion_params_t *)malloc(sizeof(dt_iop_filminversion_params_t));
    memcpy(n, self->default_params, sizeof(dt_iop_filminversion_params_t));
    *new_params = n;
    *new_params_size = sizeof(dt_iop_filminversion_params_t);
    *new_version = 8;
    return 0;
  }

  // Version 5 -> 6: dye_separation and separation_damping were added at the
  // end of the params struct. Copy the old fields and default the new ones.
  if(old_version == 5)
  {
    typedef struct dt_iop_filminversion_params_v5_t
    {
      dt_iop_filminversion_mode_t mode;
      float mask_offset[4];
      float filtration[4];
      float k_trim[4];
      float cast_removal_strength;
      float density;
      float shadow_density;
      float highlight_density;
      float grade;
      float shadow_grade;
      float highlight_grade;
      float snap;
      float toe;
      float shoulder;
      float toe_width;
      float shoulder_width;
      float exposure;
    } dt_iop_filminversion_params_v5_t;

    dt_iop_filminversion_params_v5_t *o = (dt_iop_filminversion_params_v5_t *)old_params;
    dt_iop_filminversion_params_t *n = (dt_iop_filminversion_params_t *)malloc(sizeof(dt_iop_filminversion_params_t));
    memcpy(n, o, sizeof(dt_iop_filminversion_params_v5_t));
    n->dye_separation = 0.0f;
    n->separation_damping = 0.0f;
    *new_params = n;
    *new_params_size = sizeof(dt_iop_filminversion_params_t);
    *new_version = 6;
    return 0;
  }

  // Version 6 -> 7: shadow_filtration and highlight_filtration (Regional CMY)
  // were added at the end of the params struct. Copy the old fields and
  // default the new ones.
  if(old_version == 6)
  {
    typedef struct dt_iop_filminversion_params_v6_t
    {
      dt_iop_filminversion_mode_t mode;
      float mask_offset[4];
      float filtration[4];
      float k_trim[4];
      float cast_removal_strength;
      float density;
      float shadow_density;
      float highlight_density;
      float grade;
      float shadow_grade;
      float highlight_grade;
      float snap;
      float toe;
      float shoulder;
      float toe_width;
      float shoulder_width;
      float exposure;
      float dye_separation;
      float separation_damping;
    } dt_iop_filminversion_params_v6_t;

    dt_iop_filminversion_params_v6_t *o = (dt_iop_filminversion_params_v6_t *)old_params;
    dt_iop_filminversion_params_t *n = (dt_iop_filminversion_params_t *)malloc(sizeof(dt_iop_filminversion_params_t));
    memcpy(n, o, sizeof(dt_iop_filminversion_params_v6_t));
    for(int c = 0; c < 4; c++) { n->shadow_filtration[c] = 0.0f; n->highlight_filtration[c] = 0.0f; }
    *new_params = n;
    *new_params_size = sizeof(dt_iop_filminversion_params_t);
    *new_version = 7;
    return 0;
  }

  // Version 7 -> 8: per-channel trims (toe/shoulder height/width, snap, dye
  // separation) were added at the end of the params struct. Copy the old
  // fields and default the new ones.
  if(old_version == 7)
  {
    typedef struct dt_iop_filminversion_params_v7_t
    {
      dt_iop_filminversion_mode_t mode;
      float mask_offset[4];
      float filtration[4];
      float k_trim[4];
      float cast_removal_strength;
      float density;
      float shadow_density;
      float highlight_density;
      float grade;
      float shadow_grade;
      float highlight_grade;
      float snap;
      float toe;
      float shoulder;
      float toe_width;
      float shoulder_width;
      float exposure;
      float dye_separation;
      float separation_damping;
      float shadow_filtration[4];
      float highlight_filtration[4];
    } dt_iop_filminversion_params_v7_t;

    dt_iop_filminversion_params_v7_t *o = (dt_iop_filminversion_params_v7_t *)old_params;
    dt_iop_filminversion_params_t *n = (dt_iop_filminversion_params_t *)malloc(sizeof(dt_iop_filminversion_params_t));
    memcpy(n, o, sizeof(dt_iop_filminversion_params_v7_t));
    for(int c = 0; c < 4; c++)
    {
      n->toe_trim[c] = 0.0f;
      n->shoulder_trim[c] = 0.0f;
      n->toe_width_trim[c] = 0.0f;
      n->shoulder_width_trim[c] = 0.0f;
      n->snap_trim[c] = 0.0f;
      n->dye_separation_trim[c] = 0.0f;
    }
    *new_params = n;
    *new_params_size = sizeof(dt_iop_filminversion_params_t);
    *new_version = 8;
    return 0;
  }

  return 1;
}

// Numerically stable softplus: log(1 + exp(x))
static inline float _softplus(float x)
{
  return x > 0.0f ? x + logf(1.0f + expf(-x)) : logf(1.0f + expf(x));
}

// 10^-D, as expf (markedly cheaper than powf in the per-pixel loop).
static inline float _density_to_linear(float density)
{
  return expf(-density * (float)M_LN10);
}

// Black point compensation: remaps the paper's physical black transmittance
// (bpc_black, see _compute_curve_shape()) up to true 0, rescaling so paper
// white stays near 1 -- an ICC relative-colorimetric-style remap. Applied
// unconditionally, matching NegPy's own default (bpc = not paper_black,
// paper_black defaults to off).
static inline float _apply_bpc(float linear, float bpc_black)
{
  return (linear - bpc_black) / (1.0f - bpc_black);
}

// v is k * x_adj (slope-scaled log exposure, pivot at v=0); returns print
// density, not yet converted back to linear.
static inline float _hd_curve(float v, float a_sh, float a_hl, float d_min_eff, float d_max_eff)
{
  const float v1 = d_min_eff + _softplus(a_hl * (v - d_min_eff)) / a_hl;  // shoulder -> paper white
  return d_max_eff - _softplus(a_sh * (d_max_eff - v1)) / a_sh;           // toe -> paper black
}

// Numerically stable log(exp(y) - 1), for y > 0.
static inline float _inv_softplus(float y)
{
  return y > 20.0f ? y : logf(expm1f(fmaxf(y, 1e-6f)));
}

// Closed-form solve (double inverse-softplus) for the v_star where
// _hd_curve() == HD_ANCHOR_TARGET_DENSITY, independent of grade/toe/shoulder.
static inline float _solve_v_star(float a_sh, float a_hl, float d_min_eff, float d_max_eff)
{
  const float target = HD_ANCHOR_TARGET_DENSITY;
  const float v1 = d_max_eff - _inv_softplus(a_sh * (d_max_eff - target)) / a_sh;
  return d_min_eff + _inv_softplus(a_hl * (v1 - d_min_eff)) / a_hl;
}

// Split Grade trim as a gain (multiplier - 1), applied additively in
// proportion to distance from the zone center -- so the center itself is
// preserved and only the local slope rotates around it.
static inline float _grade_trim_gain(float grade, float trim)
{
  const float r0 = fmaxf(105.0f, fminf(150.0f, grade));
  const float r1 = fmaxf(105.0f, fminf(150.0f, r0 + trim));
  return r0 / r1 - 1.0f;
}

// Temperature lever, GUI-only (not used by process()/commit_params -- these
// convert between the stored filtration magenta/yellow pair and a Kelvin
// display/lever value, ported from NegPy's wb_to_kelvin()/kelvin_to_wb()).
//
// Measured print temperature: least-squares projection of the (magenta,
// yellow) pair onto the Planckian (mired) direction; HD_TEMP_REF_KELVIN at
// neutral.
static inline float _wb_to_kelvin(float magenta, float yellow)
{
  const float km = HD_TEMP_K_MAGENTA, ky = HD_TEMP_K_YELLOW;
  const float dmu = (km * magenta + ky * yellow) / (km * km + ky * ky);
  // clamp in the mired domain: a Kelvin-domain clamp breaks when mu goes negative
  const float mu = fmaxf(1e6f / HD_TEMP_MAX_KELVIN, fminf(1e6f / HD_TEMP_MIN_KELVIN, 1e6f / HD_TEMP_REF_KELVIN + dmu));
  return 1e6f / mu;
}

// Moves (magenta, yellow) along the Planckian direction to land on `kelvin`,
// keeping the off-locus green-magenta tint component untouched. Clips to
// the [-1,1] slider range.
static inline void _kelvin_to_wb(float kelvin, float magenta, float yellow, float *magenta_out, float *yellow_out)
{
  const float km = HD_TEMP_K_MAGENTA, ky = HD_TEMP_K_YELLOW;
  const float kelvin_c = fmaxf(HD_TEMP_MIN_KELVIN, fminf(HD_TEMP_MAX_KELVIN, kelvin));
  const float dmu_cur = (km * magenta + ky * yellow) / (km * km + ky * ky);
  const float d = (1e6f / kelvin_c - 1e6f / HD_TEMP_REF_KELVIN) - dmu_cur;
  *magenta_out = fmaxf(-1.0f, fminf(1.0f, magenta + km * d));
  *yellow_out = fmaxf(-1.0f, fminf(1.0f, yellow + ky * d));
}

// Density-domain saturation + Separation Damping. Operates on the print
// curve's output density in-place; base is the per-channel paper-white
// floor, k the per-channel separation gain (NegPy per_channel_dye_separation
// trims). k=1 is identity, k>1 boosts density separation. damping=0 gives
// the constant matrix k*I + (1-k)*J per channel; damping>0 makes each
// channel's k chroma-dependent using NegPy's separation_damping_gain law
// (chroma itself is shared -- one property of the pixel, not per-channel).
// Neutrals are preserved because the transform only touches deviations from
// the mean. Caller gates on "is any k[c] > 1" -- this always computes since
// checking a whole array cheaply isn't worth complicating the call site for.
static inline void _apply_dye_separation(float *const restrict D,
                                         const dt_aligned_pixel_t base,
                                         const dt_aligned_pixel_t k,
                                         const float damping)
{
  float e[3];
  for(int c = 0; c < 3; c++) e[c] = D[c] - base[c];
  const float e_mean = (e[0] + e[1] + e[2]) / 3.0f;

  if(damping <= 0.0f)
  {
    for(int c = 0; c < 3; c++) D[c] = base[c] + e_mean + k[c] * (e[c] - e_mean);
    return;
  }

  const float d0 = e[0] - e_mean;
  const float d1 = e[1] - e_mean;
  const float d2 = e[2] - e_mean;
  const float chroma = sqrtf(((d0 - d1) * (d0 - d1) + (d1 - d2) * (d1 - d2) + (d0 - d2) * (d0 - d2)) / 3.0f);
  const float h = (HD_SEPARATION_DAMPING_REF_SPREAD - chroma) / fmaxf(HD_SEPARATION_DAMPING_REF_SPREAD + chroma, 1e-6f);
  const float exponent = (1.0f - damping) + damping * h;

  for(int c = 0; c < 3; c++)
  {
    const float gain = fminf(powf(k[c], exponent), 3.0f);
    D[c] = base[c] + e_mean + gain * (e[c] - e_mean);
  }
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_filminversion_params_t *const p = (dt_iop_filminversion_params_t *)p1;
  dt_iop_filminversion_data_t *const d = piece->data;

  d->mode = p->mode;
  for(size_t c = 0; c < 4; c++) d->mask_offset[c] = p->mask_offset[c];
  for(size_t c = 0; c < 4; c++) d->filtration[c] = p->filtration[c];
  for(size_t c = 0; c < 4; c++) d->k_trim[c] = fmaxf(0.5f, fminf(2.0f, p->k_trim[c]));
  for(size_t c = 0; c < 4; c++) d->shadow_filtration[c] = p->shadow_filtration[c] * HD_CMY_MAX_DENSITY;
  for(size_t c = 0; c < 4; c++) d->highlight_filtration[c] = p->highlight_filtration[c] * HD_CMY_MAX_DENSITY;
  d->cast_removal_strength = fmaxf(-1.0f, fminf(1.0f, p->cast_removal_strength));
  d->density = p->density;
  d->exposure = p->exposure;
  d->shadow_density = p->shadow_density;
  d->highlight_density = p->highlight_density;
  d->midtone_gamma = HD_PAPER_MIDTONE_GAMMA + p->snap;
  // Clamp the final k (NegPy per_channel_dye_separation's own [0,3] range),
  // not (dye_separation + trim) against the *global* slider's [0,1] domain
  // first -- that earlier clamp floored at 0 before the +1.0 ever happened,
  // so with dye_separation at its own default minimum (0), any negative
  // trim was silently zeroed and could never do anything.
  for(size_t c = 0; c < 4; c++)
    d->dye_k[c] = fmaxf(0.0f, fminf(3.0f, 1.0f + p->dye_separation + p->dye_separation_trim[c]));
  d->separation_damping = fmaxf(0.0f, fminf(1.0f, p->separation_damping));
  for(size_t c = 0; c < 4; c++) d->snap_trim[c] = p->snap_trim[c];

  // The slider is inverted relative to the physical ISO-R scale (higher raw
  // value = harder) and narrowed to [105,150], the only band where the slope
  // isn't already pinned to HD_SLOPE_MIN for a typical density_range --
  // mirror it around the midpoint to recover the physical value.
  const float grade_raw = fmaxf(105.0f, fminf(150.0f, p->grade));
  const float grade = 105.0f + 150.0f - grade_raw;
  // k = grade_contrast_scale * density_range / (grade/100); density_range is
  // per-frame, so only the grade-dependent factor is precomputed here.
  d->grade_slope_factor = HD_GRADE_CONTRAST_SCALE / (grade / 100.0f);
  d->shadow_grade_gain = _grade_trim_gain(grade, p->shadow_grade);
  d->highlight_grade_gain = _grade_trim_gain(grade, p->highlight_grade);

  // Toe/shoulder shape is grade-coupled (see _compute_curve_shape()), which
  // needs the per-frame measured slope -- unavailable here, so only the raw
  // slider inputs are stashed for _frame_from_measurement() to solve later.
  d->toe = p->toe;
  d->shoulder = p->shoulder;
  d->toe_width = fmaxf(p->toe_width, 0.1f);
  d->shoulder_width = fmaxf(p->shoulder_width, 0.1f);
  for(size_t c = 0; c < 4; c++) d->toe_trim[c] = p->toe_trim[c];
  for(size_t c = 0; c < 4; c++) d->shoulder_trim[c] = p->shoulder_trim[c];
  for(size_t c = 0; c < 4; c++) d->toe_width_trim[c] = p->toe_width_trim[c];
  for(size_t c = 0; c < 4; c++) d->shoulder_width_trim[c] = p->shoulder_width_trim[c];
}

// Solves a_sh/a_hl/d_min_eff/d_max_eff/bpc_black for one already-final toe/
// shoulder height+width (post grade-coupling, post per-channel trim, post
// clamp -- see _compute_curve_shape(), which is the only caller). Shared by
// both the per-channel solve and the B&W/pivot-basis solve below.
static void _curve_shape_solve(float toe_final, float shoulder_final,
                               float toe_width_final, float shoulder_width_final,
                               float *const a_sh, float *const a_hl,
                               float *const d_min_eff, float *const d_max_eff, float *const bpc_black)
{
  const float toe = toe_final * HD_TOE_SHOULDER_STRENGTH;
  const float shoulder = shoulder_final * HD_TOE_SHOULDER_STRENGTH;

  // toe: positive values lift the paper-black ceiling (fog shadows, matches
  // real paper toe behavior); negative values sharpen the shadow knee instead.
  if(toe >= 0.0f)
  {
    *d_max_eff = HD_D_MAX - toe * HD_TOE_HEIGHT;
    *a_sh = HD_TOE_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF / toe_width_final;
  }
  else
  {
    *d_max_eff = HD_D_MAX;
    *a_sh = (HD_TOE_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF / toe_width_final) * (1.0f - toe * 4.0f);
  }

  // Black point compensation clip point (NegPy bpc_black): references the
  // *physical* d_max, not d_max_eff, so a positive toe's lifted black still
  // gets remapped to true black. Negative toe is the exception -- it sharpens
  // the knee rather than lifting it, so the curve only asymptotically
  // approaches d_max and the clip point has to move inward to land somewhere
  // the curve can actually reach.
  {
    float db = HD_D_MAX;
    if(toe < 0.0f) db = HD_D_MAX + toe * HD_TOE_HEIGHT;
    *bpc_black = _density_to_linear(db);
  }

  // shoulder: positive values lift the paper-white floor (compress
  // highlights toward grey); negative values sharpen the highlight knee.
  if(shoulder >= 0.0f)
  {
    *d_min_eff = HD_D_MIN + shoulder * HD_SHOULDER_HEIGHT;
    *a_hl = HD_SHOULDER_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF / shoulder_width_final;
  }
  else
  {
    *d_min_eff = HD_D_MIN;
    *a_hl = (HD_SHOULDER_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF / shoulder_width_final) * (1.0f - shoulder * 4.0f);
  }

  if(*d_max_eff < *d_min_eff + 0.1f) *d_max_eff = *d_min_eff + 0.1f;
}

// Toe/shoulder curve shape, grade-coupled (NegPy grade_coupled_shape) and
// per-channel trimmed (NegPy per_channel_toe_shoulder/per_channel_widths):
// hard grades (high slope) physically have snappier toes and more
// compressed shoulders than soft ones, so slope_norm (0 = softest grade, 1 =
// hardest) nudges the effective toe/shoulder height globally, then each
// channel's own trim nudges it further, clamped to the same [-1,1]/[0.1,5.0]
// domain the global sliders themselves are clamped to. This is why the curve
// solve can't live in commit_params() alongside the rest of the shape math:
// base_slope is only known once the per-frame density_range has been
// measured.
//
// Produces both the per-channel shape (a_sh/a_hl/d_min_eff/d_max_eff/
// bpc_black, used by C41/E6) and a shared scalar version (used by B&W, which
// has no per-channel trims). v_star/anchor_scale are solved from the shared
// scalar version regardless of mode -- matching NegPy's
// _reference_linear_value(), which uses only the base sharpness constants
// and is channel-independent by construction, not swayed by any one
// channel's trim.
static void _compute_curve_shape(const dt_iop_filminversion_data_t *const d, float base_slope,
                                 dt_aligned_pixel_t a_sh, dt_aligned_pixel_t a_hl,
                                 dt_aligned_pixel_t d_min_eff, dt_aligned_pixel_t d_max_eff,
                                 dt_aligned_pixel_t bpc_black,
                                 float *const a_sh_bw, float *const a_hl_bw,
                                 float *const d_min_eff_bw, float *const d_max_eff_bw, float *const bpc_black_bw,
                                 float *const v_star, float *const anchor_scale)
{
  const float slope_norm = fmaxf(0.0f, fminf(1.0f, (base_slope - HD_SLOPE_MIN) / (HD_SLOPE_MAX - HD_SLOPE_MIN)));
  const float toe_eff = d->toe + HD_TOE_GRADE_STRENGTH * slope_norm;
  const float shoulder_eff = d->shoulder + HD_SHOULDER_GRADE_STRENGTH * slope_norm;

  _curve_shape_solve(toe_eff, shoulder_eff, d->toe_width, d->shoulder_width,
                     a_sh_bw, a_hl_bw, d_min_eff_bw, d_max_eff_bw, bpc_black_bw);

  for(int c = 0; c < 3; c++)
  {
    const float toe_c = fmaxf(-1.0f, fminf(1.0f, toe_eff + d->toe_trim[c]));
    const float shoulder_c = fmaxf(-1.0f, fminf(1.0f, shoulder_eff + d->shoulder_trim[c]));
    const float toe_width_c = fmaxf(0.1f, fminf(5.0f, d->toe_width + d->toe_width_trim[c]));
    const float shoulder_width_c = fmaxf(0.1f, fminf(5.0f, d->shoulder_width + d->shoulder_width_trim[c]));
    _curve_shape_solve(toe_c, shoulder_c, toe_width_c, shoulder_width_c,
                       &a_sh[c], &a_hl[c], &d_min_eff[c], &d_max_eff[c], &bpc_black[c]);
  }

  // Pivot solved onto HD_ANCHOR_TARGET_DENSITY, then that fixed density
  // anchored to 18% mid-gray -- so anchor_scale never drifts with grade.
  *v_star = _solve_v_star(*a_sh_bw, *a_hl_bw, *d_min_eff_bw, *d_max_eff_bw);
  *anchor_scale = 0.18f / _density_to_linear(HD_ANCHOR_TARGET_DENSITY);
}

// In-place quickselect (Lomuto, iterative, median-of-three): returns the
// k-th smallest of arr[0..n), leaving arr partitioned but not sorted. The
// measurements below only need order statistics, so this is O(n) average
// instead of a full O(n log n) sort of a grid with many thousands of cells.
static float _quickselect_float(float *arr, size_t n, size_t k)
{
  size_t left = 0, right = n - 1;
  while(left < right)
  {
    const size_t mid = left + (right - left) / 2;
    if(arr[mid] < arr[left]) { const float t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; }
    if(arr[right] < arr[left]) { const float t = arr[left]; arr[left] = arr[right]; arr[right] = t; }
    if(arr[right] < arr[mid]) { const float t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; }
    const float pivot = arr[mid];

    // Three-way (Dutch national flag) partition around `pivot`: [left,lo) < pivot,
    // [lo,hi] == pivot, (hi,right] > pivot. A plain two-way partition (as this used
    // to be) only ever peels the single pivot value off per pass when many elements
    // are equal to it, degrading to O(n^2) -- exactly what a flat, low-texture scan's
    // density grid triggers (long runs of identical/near-identical values). Grouping
    // the equal band restores real, guaranteed progress every pass regardless of how
    // many duplicates there are.
    size_t lo = left, hi = right, i = left;
    while(i <= hi)
    {
      if(arr[i] < pivot)
      {
        const float t = arr[i]; arr[i] = arr[lo]; arr[lo] = t;
        lo++; i++;
      }
      else if(arr[i] > pivot)
      {
        const float t = arr[i]; arr[i] = arr[hi]; arr[hi] = t;
        if(hi == left) break; // would underflow; nothing left below to scan
        hi--;
      }
      else i++;
    }

    if(k >= lo && k <= hi) return pivot;
    else if(k < lo) right = (lo == 0) ? 0 : lo - 1;
    else left = hi + 1;
  }
  return arr[left];
}

// Drops the unexposed surround from the grid, compacting the survivors to
// the front of grid[0..2] and returning their count.
//
// A cell qualifies only if all three hold: it has no local texture, it is
// reachable from the frame border through other such cells, and its density
// is far from the median of the textured cells. Border connectivity is what
// preserves genuinely flat picture content -- skies, walls, studio backdrops
// -- and taking the reference from textured cells alone keeps the separation
// test honest even when the surround covers much of the frame.
//
// Bails out with the grid untouched whenever the outcome would be
// untrustworthy, so a frame with no surround, or one flat throughout, is
// left exactly as it was.
static size_t _reject_unexposed_border(float *const grid[3], const int gw, const int gh)
{
  const size_t ncells = (size_t)gw * (size_t)gh;
  if(gw < 3 || gh < 3 || ncells < (size_t)HD_REBATE_MIN_CELLS) return ncells;

  float *const luma = malloc(sizeof(float) * ncells);
  float *const detail = malloc(sizeof(float) * ncells);
  if(!luma || !detail) { free(luma); free(detail); return ncells; }

  for(size_t i = 0; i < ncells; i++)
    luma[i] = 0.2126f * grid[0][i] + 0.7152f * grid[1][i] + 0.0722f * grid[2][i];

  // Local texture as the mean absolute difference to the 4-neighbourhood.
  // Taken across the grid rather than within a block so it stays meaningful
  // when blocks are a single pixel wide, as they are on the preview pipe.
  DT_OMP_FOR()
  for(int gy = 0; gy < gh; gy++)
  {
    for(int gx = 0; gx < gw; gx++)
    {
      const size_t i = (size_t)gy * gw + gx;
      float acc = 0.0f;
      int n = 0;
      if(gx > 0)      { acc += fabsf(luma[i] - luma[i - 1]);  n++; }
      if(gx < gw - 1) { acc += fabsf(luma[i] - luma[i + 1]);  n++; }
      if(gy > 0)      { acc += fabsf(luma[i] - luma[i - gw]); n++; }
      if(gy < gh - 1) { acc += fabsf(luma[i] - luma[i + gw]); n++; }
      detail[i] = n ? acc / (float)n : 0.0f;
    }
  }

  float *const tex = malloc(sizeof(float) * ncells);
  if(!tex) { free(detail); free(luma); return ncells; }

  size_t ntex = 0;
  for(size_t i = 0; i < ncells; i++)
    if(detail[i] >= HD_REBATE_FLATNESS) tex[ntex++] = luma[i];

  if(ntex < (size_t)HD_REBATE_MIN_CELLS)
  {
    free(tex); free(detail); free(luma);
    return ncells;
  }
  const float content = _quickselect_float(tex, ntex, ntex / 2);
  free(tex);

  uint8_t *const reject = calloc(ncells, sizeof(uint8_t));
  uint32_t *const stack = malloc(sizeof(uint32_t) * ncells);
  if(!reject || !stack)
  {
    free(stack); free(reject); free(detail); free(luma);
    return ncells;
  }

  size_t top = 0;
  for(int gy = 0; gy < gh; gy++)
  {
    const gboolean edge_row = (gy == 0 || gy == gh - 1);
    for(int gx = 0; gx < gw; gx++)
    {
      if(!edge_row && gx != 0 && gx != gw - 1) continue;
      const size_t i = (size_t)gy * gw + gx;
      if(detail[i] < HD_REBATE_FLATNESS && fabsf(luma[i] - content) > HD_REBATE_SEPARATION)
      {
        reject[i] = 1;
        stack[top++] = (uint32_t)i;
      }
    }
  }

  while(top)
  {
    const size_t i = stack[--top];
    const int gx = (int)(i % (size_t)gw);
    const int gy = (int)(i / (size_t)gw);
    const int nx[4] = { gx - 1, gx + 1, gx, gx };
    const int ny[4] = { gy, gy, gy - 1, gy + 1 };
    for(int k = 0; k < 4; k++)
    {
      if(nx[k] < 0 || nx[k] >= gw || ny[k] < 0 || ny[k] >= gh) continue;
      const size_t j = (size_t)ny[k] * gw + nx[k];
      if(reject[j] || detail[j] >= HD_REBATE_FLATNESS) continue;
      if(fabsf(luma[j] - content) <= HD_REBATE_SEPARATION) continue;
      reject[j] = 1;
      stack[top++] = (uint32_t)j;
    }
  }

  size_t kept = 0;
  for(size_t i = 0; i < ncells; i++) if(!reject[i]) kept++;

  if(kept < (size_t)HD_REBATE_MIN_CELLS || (float)kept < HD_REBATE_MIN_KEEP * (float)ncells)
    kept = ncells; // too little would survive to measure from -- keep it all
  else
  {
    size_t w = 0;
    for(size_t i = 0; i < ncells; i++)
    {
      if(reject[i]) continue;
      for(int c = 0; c < 3; c++) grid[c][w] = grid[c][i];
      w++;
    }
    kept = w;
  }

  free(stack);
  free(reject);
  free(detail);
  free(luma);
  return kept;
}

// Block-median density grid shared by the floor/ceiling, textural-range and
// neutral-axis measurements. The per-block median of a small fixed sample is
// what makes the statistics robust to dust specks and grain. The unexposed
// surround is rejected before returning, so the count returned may be lower
// than the grid dimensions imply. Caller frees grid_out[0..2] via
// dt_free_align().
static size_t _build_density_grid(const float *const restrict in, const int width, const int height,
                                  float *grid_out[3])
{
  // Analysis area is [ax0,ax1) x [ay0,ay1), see HD_ANALYSIS_EDGE_MARGIN.
  const int mx = (int)(width * HD_ANALYSIS_EDGE_MARGIN);
  const int my = (int)(height * HD_ANALYSIS_EDGE_MARGIN);
  const int ax0 = mx, ay0 = my;
  const int ax1 = width - mx > ax0 ? width - mx : width;
  const int ay1 = height - my > ay0 ? height - my : height;
  const int aw = ax1 - ax0;
  const int ah = ay1 - ay0;

  const int longer = aw > ah ? aw : ah;
  int block = longer / HD_GRID_TARGET;
  if(block < 1) block = 1;
  const int gw = aw / block < 1 ? 1 : aw / block;
  const int gh = ah / block < 1 ? 1 : ah / block;
  const size_t ncells = (size_t)gw * (size_t)gh;

  for(int c = 0; c < 3; c++) grid_out[c] = dt_alloc_align_float(ncells);

  DT_OMP_FOR()
  for(int gy = 0; gy < gh; gy++)
  {
    for(int gx = 0; gx < gw; gx++)
    {
      const int x0 = ax0 + gx * block;
      const int y0 = ay0 + gy * block;
      const int x1 = (x0 + block < ax1) ? x0 + block : ax1;
      const int y1 = (y0 + block < ay1) ? y0 + block : ay1;

      // fixed 5x5 sample per block -> cheap approximate median
      float sample[3][25];
      int nsamp = 0;
      for(int sy = 0; sy < 5 && nsamp < 25; sy++)
      {
        const int py = y0 + (sy * (y1 - y0)) / 5;
        for(int sx = 0; sx < 5 && nsamp < 25; sx++)
        {
          const int px = x0 + (sx * (x1 - x0)) / 5;
          const float *const pix = in + ((size_t)py * width + (size_t)px) * 4;
          for(int c = 0; c < 3; c++)
            sample[c][nsamp] = -log10f(fmaxf(pix[c], DENSITY_EPS));
          nsamp++;
        }
      }

      for(int c = 0; c < 3; c++)
      {
        for(int i = 1; i < nsamp; i++)  // insertion sort, nsamp <= 25
        {
          const float key = sample[c][i];
          int j = i - 1;
          while(j >= 0 && sample[c][j] > key) { sample[c][j + 1] = sample[c][j]; j--; }
          sample[c][j + 1] = key;
        }
        grid_out[c][(size_t)gy * gw + gx] = sample[c][nsamp / 2];
      }
    }
  }
  return _reject_unexposed_border(grid_out, gw, gh);
}

// This IS the mask neutralization: it absorbs the orange mask's per-channel
// density offset automatically. Copies before selecting, since the caller
// reuses grid_in for the neutral-axis measurement afterwards.
static void _percentile_floor_ceiling(float *const grid_in[3], size_t ncells,
                                      dt_aligned_pixel_t floor_out, dt_aligned_pixel_t ceil_out)
{
  float *const tmp = dt_alloc_align_float(ncells);
  for(int c = 0; c < 3; c++)
  {
    memcpy(tmp, grid_in[c], sizeof(float) * ncells);
    const size_t idx_floor = (size_t)(HD_FLOOR_PERCENTILE / 100.0f * (ncells - 1));
    const size_t idx_ceil = (size_t)(HD_CEIL_PERCENTILE / 100.0f * (ncells - 1));
    floor_out[c] = _quickselect_float(tmp, ncells, idx_floor);
    ceil_out[c] = _quickselect_float(tmp, ncells, idx_ceil);
    if(ceil_out[c] - floor_out[c] < 1e-3f) ceil_out[c] = floor_out[c] + 1e-3f;
  }
  dt_free_align(tmp);
  floor_out[3] = 0.0f;
  ceil_out[3] = 1.0f;
}

// P10-P90 spread of the RAW (not floor/ceiling-normalized) luma-weighted
// log-density grid, in absolute log10 units. Clamping happens later, in
// _effective_grade_range().
static float _measure_textural_range(float *const grid[3], size_t ncells)
{
  float *const lum = malloc(sizeof(float) * ncells);
  for(size_t i = 0; i < ncells; i++)
    lum[i] = 0.2126f * grid[0][i] + 0.7152f * grid[1][i] + 0.0722f * grid[2][i];
  const size_t idx_lo = (size_t)(HD_TEXTURAL_LO_PERCENTILE / 100.0f * (ncells - 1));
  const size_t idx_hi = (size_t)(HD_TEXTURAL_HI_PERCENTILE / 100.0f * (ncells - 1));
  const float v_lo = _quickselect_float(lum, ncells, idx_lo);
  const float v_hi = _quickselect_float(lum, ncells, idx_hi);
  const float rng = fabsf(v_hi - v_lo);
  free(lum);
  return rng;
}

// Luma-weighted per-channel floor-to-ceiling spread, in absolute log10
// units -- the other input to _effective_grade_range() below.
static inline float _floor_ceil_range(const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil)
{
  return 0.2126f * fabsf(ceil[0] - floor[0]) + 0.7152f * fabsf(ceil[1] - floor[1]) + 0.0722f * fabsf(ceil[2] - floor[2]);
}

// Damps the floor_ceil/textural ratio toward a canonical "normal negative"
// ratio, rather than feeding either measured range straight into the slope
// calibration. Always applied (no toggle), see HD_AUTO_GRADE_*.
static float _effective_grade_range(float floor_ceil_range, float textural_range)
{
  if(textural_range < 1e-6f) return HD_TEXTURAL_RANGE_MAX; // degenerate (near-flat) frame
  const float ratio = fabsf(floor_ceil_range) / textural_range;
  return HD_AUTO_GRADE_TARGET * (HD_AUTO_GRADE_NOMINAL_RATIO + HD_AUTO_GRADE_STRENGTH * (ratio - HD_AUTO_GRADE_NOMINAL_RATIO));
}

// The frame's own median luma-weighted normalized value, blended a clamped
// fraction into HD_ASSUMED_ANCHOR.
static float _measure_anchor(float *const grid[3], size_t ncells,
                             const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil)
{
  float *const lum = malloc(sizeof(float) * ncells);
  for(size_t i = 0; i < ncells; i++)
  {
    float norm[3];
    for(int c = 0; c < 3; c++)
    {
      const float range = fmaxf(ceil[c] - floor[c], 1e-3f);
      norm[c] = (grid[c][i] - floor[c]) / range;
    }
    lum[i] = 0.2126f * norm[0] + 0.7152f * norm[1] + 0.0722f * norm[2];
  }
  const size_t idx = (size_t)(HD_ANCHOR_METER_PERCENTILE / 100.0f * (ncells - 1));
  const float measured = _quickselect_float(lum, ncells, idx);
  free(lum);

  const float anchor = HD_ASSUMED_ANCHOR + HD_ANCHOR_METER_STRENGTH * (measured - HD_ASSUMED_ANCHOR);
  return fmaxf(HD_ASSUMED_ANCHOR - HD_ANCHOR_METER_BAND, fminf(HD_ASSUMED_ANCHOR + HD_ANCHOR_METER_BAND, anchor));
}

// Normalizes a raw log-density reference into the curve's [0,1]-ish domain.
static inline void _normalize_ref(const dt_aligned_pixel_t ref, const dt_aligned_pixel_t floor,
                                  const dt_aligned_pixel_t ceil, dt_aligned_pixel_t out)
{
  for(int c = 0; c < 3; c++)
  {
    const float range = fmaxf(ceil[c] - floor[c], 1e-3f);
    out[c] = (ref[c] - floor[c]) / range;
  }
}

// Rotation-symmetric chroma distance from the neutral axis.
static inline float _rms_chroma3(float r, float g, float b)
{
  const float rg = r - g, gb = g - b, rb = r - b;
  return sqrtf((rg * rg + gb * gb + rb * rb) / 3.0f);
}

typedef struct
{
  gboolean found;
  dt_aligned_pixel_t refs; // per-channel median RAW log density of the selected set
  float chroma;            // median chroma of the selected set
  int count;
} _band_result_t;

typedef struct { float chroma; size_t idx; } _chroma_idx_t;

// _quickselect_float, partitioning by .chroma.
static float _quickselect_chroma(_chroma_idx_t *arr, size_t n, size_t k)
{
  size_t left = 0, right = n - 1;
  while(left < right)
  {
    const size_t mid = left + (right - left) / 2;
    if(arr[mid].chroma < arr[left].chroma) { const _chroma_idx_t t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; }
    if(arr[right].chroma < arr[left].chroma) { const _chroma_idx_t t = arr[left]; arr[left] = arr[right]; arr[right] = t; }
    if(arr[right].chroma < arr[mid].chroma) { const _chroma_idx_t t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; }
    const float pivot = arr[mid].chroma;

    // See _quickselect_float(): three-way partition so long runs of equal
    // chroma values (common once a lot of grid cells pass the near-neutral
    // filter) don't degrade this to O(n^2).
    size_t lo = left, hi = right, i = left;
    while(i <= hi)
    {
      if(arr[i].chroma < pivot)
      {
        const _chroma_idx_t t = arr[i]; arr[i] = arr[lo]; arr[lo] = t;
        lo++; i++;
      }
      else if(arr[i].chroma > pivot)
      {
        const _chroma_idx_t t = arr[i]; arr[i] = arr[hi]; arr[hi] = t;
        if(hi == left) break; // would underflow; nothing left below to scan
        hi--;
      }
      else i++;
    }

    if(k >= lo && k <= hi) return pivot;
    else if(k < lo) right = (lo == 0) ? 0 : lo - 1;
    else left = hi + 1;
  }
  return arr[left].chroma;
}

// Among grid cells whose luma is in [lo, hi], keeps the lowest-chroma
// HD_CAST_CHROMA_QUANTILE fraction as the near-neutral set (rejecting the
// band if that set is too small or still too chromatic under `cap`), and
// returns its per-channel median raw log density.
static _band_result_t _band_refs(float *const grid[3], const float *const luma,
                                 const float *const chroma, size_t ncells,
                                 float lo, float hi, float cap)
{
  _band_result_t res = { .found = FALSE, .chroma = 1.0f, .count = 0 };

  _chroma_idx_t *const pairs = malloc(sizeof(_chroma_idx_t) * ncells);
  size_t nband = 0;
  for(size_t i = 0; i < ncells; i++)
  {
    if(luma[i] >= lo && luma[i] <= hi)
    {
      pairs[nband].chroma = chroma[i];
      pairs[nband].idx = i;
      nband++;
    }
  }

  if(nband < HD_CAST_MIN_CELLS) { free(pairs); return res; }

  const size_t nsel = (size_t)(HD_CAST_CHROMA_QUANTILE * (float)nband);
  if(nsel < HD_CAST_MIN_CELLS) { free(pairs); return res; }

  // partition the nsel smallest-chroma pairs into [0, nsel), then take their
  // median -- two O(n) selects instead of one O(n log n) sort.
  _quickselect_chroma(pairs, nband, nsel - 1);
  const float median_chroma = _quickselect_chroma(pairs, nsel, nsel / 2);
  if(median_chroma > cap) { free(pairs); return res; }

  float *const tmp = malloc(sizeof(float) * nsel);
  for(int c = 0; c < 3; c++)
  {
    for(size_t i = 0; i < nsel; i++) tmp[i] = grid[c][pairs[i].idx];
    res.refs[c] = _quickselect_float(tmp, nsel, nsel / 2);
  }
  free(tmp);
  free(pairs);

  res.chroma = median_chroma;
  res.count = (int)nsel;
  res.found = TRUE;
  return res;
}

// Two-pass chroma-gated selection of near-neutral references in the shadow/
// mid/highlight bands. Pass 1 (loose cap, mid+shadow) gives a rough R/B->G
// affine correction; pass 2 re-selects under the strict cap using that
// corrected chroma, plus an optional highlight band enabling the curvature
// fit. See HD_CAST_*_BAND for the band-direction swap vs NegPy.
static dt_iop_filminversion_neutral_axis_t _measure_neutral_axis(
    float *const grid[3], size_t ncells, const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil)
{
  dt_iop_filminversion_neutral_axis_t out = { .valid = FALSE, .has_highlight = FALSE, .confidence = 0.0f };

  float *const luma = malloc(sizeof(float) * ncells);
  float *const chroma1 = malloc(sizeof(float) * ncells);
  float *norm[3];
  for(int c = 0; c < 3; c++) norm[c] = malloc(sizeof(float) * ncells);

  for(size_t i = 0; i < ncells; i++)
  {
    for(int c = 0; c < 3; c++)
    {
      const float range = fmaxf(ceil[c] - floor[c], 1e-3f);
      norm[c][i] = (grid[c][i] - floor[c]) / range;
    }
    luma[i] = 0.2126f * norm[0][i] + 0.7152f * norm[1][i] + 0.0722f * norm[2][i];
    chroma1[i] = _rms_chroma3(norm[0][i], norm[1][i], norm[2][i]);
  }

  const _band_result_t mid1 = _band_refs(grid, luma, chroma1, ncells,
                                         HD_CAST_MID_BAND_LO, HD_CAST_MID_BAND_HI, HD_CAST_FIRST_PASS_CAP);
  const _band_result_t sh1 = _band_refs(grid, luma, chroma1, ncells,
                                        HD_CAST_SHADOW_BAND_LO, HD_CAST_SHADOW_BAND_HI, HD_CAST_FIRST_PASS_CAP);

  if(!mid1.found || !sh1.found) goto done;

  dt_aligned_pixel_t nm, ns;
  _normalize_ref(mid1.refs, floor, ceil, nm);
  _normalize_ref(sh1.refs, floor, ceil, ns);

  float a[4] = { 1.0f, 1.0f, 1.0f, 1.0f }, b[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for(int ch = 0; ch < 3; ch++)
  {
    if(ch == 1) continue; // green unchanged
    const float du = nm[ch] - ns[ch];
    if(fabsf(du) < 1e-6f) { a[ch] = 1.0f; b[ch] = nm[1] - nm[ch]; }
    else { a[ch] = (nm[1] - ns[1]) / du; b[ch] = nm[1] - a[ch] * nm[ch]; }
  }

  float *const chroma2 = malloc(sizeof(float) * ncells);
  for(size_t i = 0; i < ncells; i++)
  {
    const float cr = a[0] * norm[0][i] + b[0];
    const float cg = norm[1][i];
    const float cb = a[2] * norm[2][i] + b[2];
    chroma2[i] = _rms_chroma3(cr, cg, cb);
  }

  const _band_result_t mid2 = _band_refs(grid, luma, chroma2, ncells,
                                         HD_CAST_MID_BAND_LO, HD_CAST_MID_BAND_HI, HD_CAST_CHROMA_CAP);
  const _band_result_t sh2 = _band_refs(grid, luma, chroma2, ncells,
                                        HD_CAST_SHADOW_BAND_LO, HD_CAST_SHADOW_BAND_HI, HD_CAST_CHROMA_CAP);

  if(!mid2.found || !sh2.found) { free(chroma2); goto done; }

  const _band_result_t hl2 = _band_refs(grid, luma, chroma2, ncells,
                                        HD_CAST_HIGHLIGHT_BAND_LO, HD_CAST_HIGHLIGHT_BAND_HI, HD_CAST_CHROMA_CAP);

  for(int c = 0; c < 3; c++) { out.mid_ref[c] = mid2.refs[c]; out.shadow_ref[c] = sh2.refs[c]; }
  out.valid = TRUE;
  out.has_highlight = hl2.found;
  if(hl2.found) for(int c = 0; c < 3; c++) out.highlight_ref[c] = hl2.refs[c];

  {
    const float tight = fmaxf(0.0f, fminf(1.0f, 1.0f - fmaxf(mid2.chroma, sh2.chroma) / HD_CAST_CHROMA_CAP));
    const float size_term = (float)mid2.count / ((float)mid2.count + HD_CAST_CONFIDENCE_N0);
    dt_aligned_pixel_t dm, ds;
    _normalize_ref(mid2.refs, floor, ceil, dm);
    _normalize_ref(sh2.refs, floor, ceil, ds);
    float spread = 0.0f;
    for(int ch = 0; ch < 3; ch++)
    {
      if(ch == 1) continue;
      const float s = fabsf((dm[ch] - dm[1]) - (ds[ch] - ds[1]));
      if(s > spread) spread = s;
    }
    const float agree = 1.0f - fmaxf(0.0f, fminf(1.0f, (spread - HD_CAST_AGREEMENT_DEADZONE) / HD_CAST_AGREEMENT_SCALE));
    out.confidence = fmaxf(0.0f, fminf(1.0f, tight * size_term * agree));
  }

  free(chroma2);

done:
  for(int c = 0; c < 3; c++) free(norm[c]);
  free(luma);
  free(chroma1);
  return out;
}

// Percentile-based per-channel reference on our "shadow" (floor) side,
// unfiltered by chroma. Always available, unlike the chroma-gated neutral
// axis, so it serves as the one-point fallback.
static void _measure_shadow_tie_ref(float *const grid[3], size_t ncells, dt_aligned_pixel_t ref_out)
{
  float *const tmp = malloc(sizeof(float) * ncells);
  for(int c = 0; c < 3; c++)
  {
    memcpy(tmp, grid[c], sizeof(float) * ncells);
    const size_t idx = (size_t)(HD_CAST_SHADOW_TIE_PERCENTILE / 100.0f * (ncells - 1));
    ref_out[c] = _quickselect_float(tmp, ncells, idx);
  }
  free(tmp);
}

// Fits R/B's slope, pivot and curvature so they match green's curve at the
// near-neutral reference points, overwriting frame->slope/pivot/curv for
// R/B. Targets are built from our own green slope/pivot, keeping the solve
// self-consistent with this module's sign convention.
static void _solve_cast_removal(const dt_iop_filminversion_data_t *const d,
                                const dt_iop_filminversion_neutral_axis_t *const na,
                                float confidence, float invert_sign,
                                const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil,
                                dt_iop_filminversion_frame_t *const frame)
{
  const float strength = d->cast_removal_strength * confidence;
  // NegPy (per_channel_curve_params(), logic.py) only ever applies this
  // correction for strength > 0 -- negative values fall through to the
  // uncorrected base curve there too, there's no "exaggerate the cast"
  // branch in the reference implementation. Match that exactly.
  if(strength <= 0.0f) return;

  dt_aligned_pixel_t nm, ns, nh;
  _normalize_ref(na->mid_ref, floor, ceil, nm);
  _normalize_ref(na->shadow_ref, floor, ceil, ns);
  const gboolean have_h = na->has_highlight;
  if(have_h) _normalize_ref(na->highlight_ref, floor, ceil, nh);

  const float m_g = nm[1], s_g = ns[1];
  const float slope_g = frame->slope[1];
  const float pivot_g = frame->pivot[1];
  const float t_m = slope_g * invert_sign * (m_g - pivot_g);
  const float t_s = slope_g * invert_sign * (s_g - pivot_g);
  const float h_g = have_h ? nh[1] : 0.0f;
  const float t_h = have_h ? slope_g * invert_sign * (h_g - pivot_g) : 0.0f;

  for(int ch = 0; ch < 3; ch++)
  {
    if(ch == 1) continue; // green untouched

    // Fit position interpolates from green's own reference (strength->0+)
    // to this channel's own native reference (strength=1, matches NegPy's
    // clamp_dev()) -- a smooth, proportional blend once the slope's
    // invert_sign contamination below is corrected (that was the actual
    // cause of the earlier "any nonzero value snaps to a strong effect"
    // report, not something inherent to this fit).
    float dev_m = strength * (nm[ch] - m_g);
    dev_m = fmaxf(-HD_CAST_MIDTONE_MAX_OFFSET, fminf(HD_CAST_MIDTONE_MAX_OFFSET, dev_m));
    const float u_m = m_g + dev_m;

    float dev_s = strength * (ns[ch] - s_g);
    dev_s = fmaxf(-HD_CAST_MIDTONE_MAX_OFFSET, fminf(HD_CAST_MIDTONE_MAX_OFFSET, dev_s));
    const float u_s = s_g + dev_s;

    float curv = 0.0f;
    if(have_h)
    {
      float dev_h = strength * (nh[ch] - h_g);
      dev_h = fmaxf(-HD_CAST_MIDTONE_MAX_OFFSET, fminf(HD_CAST_MIDTONE_MAX_OFFSET, dev_h));
      const float u_h = h_g + dev_h;

      // curvature coefficient of the quadratic through (u_h,t_h), (u_m,t_m),
      // (u_s,t_s), via Cramer's rule
      const float m00 = 1.0f, m01 = u_h, m02 = u_h * u_h;
      const float m10 = 1.0f, m11 = u_m, m12 = u_m * u_m;
      const float m20 = 1.0f, m21 = u_s, m22 = u_s * u_s;
      const float det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
      if(fabsf(det) > 1e-9f)
      {
        const float det2 = m00 * (m11 * t_s - t_m * m21) - m01 * (m10 * t_s - t_m * m20) + t_h * (m10 * m21 - m11 * m20);
        curv = det2 / det;
      }
      const float curv_limit = HD_CAST_CURV_MAX_RATIO * slope_g;
      curv = fmaxf(-curv_limit, fminf(curv_limit, curv));
    }

    const float du = u_m - u_s;
    // t_m/t_s carry one factor of invert_sign (from their own definition
    // above, matching how a real v-value is computed elsewhere), which
    // divides out unevenly against du (a plain, sign-free position
    // difference) -- left uncorrected, this silently inherits that leftover
    // invert_sign factor, flipping the solved slope negative for C41
    // (invert_sign=-1) before it hits the always-positive [HD_SLOPE_MIN,
    // HD_SLOPE_MAX] clamp below, which then floors it to the wrong sign
    // entirely (observed: slope_native=1.68 clamped to exactly
    // HD_SLOPE_MIN=1.0 at strength=0.0095, a ~40% jump from a barely-on
    // slider nudge). Re-multiplying by invert_sign here cancels it back out
    // to a true, always-positive magnitude, matching frame->slope[]'s
    // convention everywhere else in this file.
    float slope_ch = (fabsf(du) < 1e-6f) ? slope_g : invert_sign * ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du;
    slope_ch = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, slope_ch * d->k_trim[ch]));
    const float curv_ch = curv * d->k_trim[ch];

    // pivot re-pin so the channel's curve passes exactly through (u_m, t_m):
    // slope_ch*invert*(u_m-pivot_ch) + curv_ch*u_m^2 = t_m
    float pivot_ch = pivot_g;
    if(fabsf(slope_ch) > 1e-6f)
      pivot_ch = u_m - invert_sign * (t_m - curv_ch * u_m * u_m) / slope_ch;

    frame->slope[ch] = slope_ch;
    frame->pivot[ch] = pivot_ch;
    frame->curv[ch] = curv_ch;
  }
}

// One-point fallback: tilts each channel's slope so its shadow-side
// reference lands on green's, pinned through the pivot's reference tone.
// Used when the neutral-axis measurement above isn't available.
static void _solve_cast_removal_fallback(const dt_iop_filminversion_data_t *const d,
                                         const dt_aligned_pixel_t shadow_tie_ref, float invert_sign,
                                         float anchor,
                                         const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil,
                                         dt_iop_filminversion_frame_t *const frame)
{
  const float strength = d->cast_removal_strength;
  // See _solve_cast_removal(): NegPy gates this branch on strength > 0.0 too
  // (logic.py, per_channel_curve_params()) -- negative values do nothing.
  if(strength <= 0.0f) return;

  dt_aligned_pixel_t r_norm;
  _normalize_ref(shadow_tie_ref, floor, ceil, r_norm);
  const float r_green = r_norm[1];
  const float numer = anchor - r_green;

  for(int ch = 0; ch < 3; ch++)
  {
    if(ch == 1) continue;

    float cast = strength * (r_green - r_norm[ch]);
    cast = fmaxf(-HD_CAST_MAX_OFFSET, fminf(HD_CAST_MAX_OFFSET, cast));
    const float denom = anchor - (r_green - cast);

    float slope_ch = frame->slope[ch];
    if(fabsf(denom) > 1e-6f)
      slope_ch = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, frame->slope[1] * numer / denom));
    slope_ch = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, slope_ch * d->k_trim[ch]));

    frame->slope[ch] = slope_ch;
    frame->pivot[ch] = anchor
                      - invert_sign * (frame->v_star / slope_ch + d->density * HD_DENSITY_MULTIPLIER);
    frame->curv[ch] = 0.0f;
  }
}

// Logistic sigmoid for the Zone Density / Split Grade mid-sparing weights.
static inline float _sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

// Applies Snap, Regional CMY, Split Grade and Zone Density to one channel's
// pre-curve v. Order matters: Snap reads the pre-everything v; Regional CMY
// (per-channel, unlike the other three) runs next, matching NegPy; Zone
// Density recomputes its weights on Split Grade's already-updated v.
// midtone_gamma is this channel's snap-trimmed value (d->midtone_gamma for
// B&W, which has no per-channel trims). shadow_cmy/highlight_cmy are this
// channel's HD_CMY_MAX_DENSITY-scaled Regional CMY value -- 0 for B&W.
static inline float _apply_snap_grade_density(float v, const dt_iop_filminversion_data_t *const d, float v_star,
                                              float midtone_gamma, float shadow_cmy, float highlight_cmy)
{
  if(midtone_gamma != 0.0f)
    v = v + midtone_gamma * HD_PAPER_GAMMA_WIDTH * tanhf((v - v_star) / HD_PAPER_GAMMA_WIDTH);

  if(shadow_cmy != 0.0f || highlight_cmy != 0.0f)
  {
    const float w_sh = _sigmoid(HD_CMY_ZONE_SHARPNESS * (v - HD_ANCHOR_TARGET_DENSITY));
    const float w_hi = 1.0f - w_sh;
    v = v + shadow_cmy * w_sh + highlight_cmy * w_hi;
  }

  if(d->shadow_grade_gain != 0.0f || d->highlight_grade_gain != 0.0f)
  {
    const float w_gsh = _sigmoid(HD_ZONE_DENSITY_SHARPNESS * (v - HD_ZONE_SHADOW_CENTER));
    const float w_ghi = 1.0f - _sigmoid(HD_ZONE_DENSITY_SHARPNESS * (v - HD_ZONE_HIGHLIGHT_CENTER));
    v = v + d->shadow_grade_gain * w_gsh * (v - HD_ZONE_SHADOW_CENTER)
          + d->highlight_grade_gain * w_ghi * (v - HD_ZONE_HIGHLIGHT_CENTER);
  }

  if(d->shadow_density != 0.0f || d->highlight_density != 0.0f)
  {
    const float w_zsh = _sigmoid(HD_ZONE_DENSITY_SHARPNESS * (v - HD_ZONE_SHADOW_CENTER));
    const float w_zhi = 1.0f - _sigmoid(HD_ZONE_DENSITY_SHARPNESS * (v - HD_ZONE_HIGHLIGHT_CENTER));
    v = v + d->shadow_density * w_zsh + d->highlight_density * w_zhi;
  }

  return v;
}

// linear -> log10 density -> per-channel normalization -> mask/filtration
// trims -> pivot/slope bridge -> Snap/Split Grade/Zone Density -> H&D curve
// -> linear -> mid-gray anchor -> anchor trim.
static inline void _process_pixel(const dt_aligned_pixel_t pix_in,
                                  dt_aligned_pixel_t pix_out,
                                  const dt_iop_filminversion_data_t *const d,
                                  const dt_iop_filminversion_frame_t *const frame)
{
  const dt_iop_filminversion_mode_t mode = d->mode;

  if(mode == DT_FILMINVERSION_BW)
  {
    // B&W: one luminance density, no mask neutralization or filtration.
    const float luma = fmaxf(0.2126f * pix_in[0] + 0.7152f * pix_in[1] + 0.0722f * pix_in[2], DENSITY_EPS);
    const float density = -log10f(luma);
    const float range = fmaxf(frame->ceil_bw - frame->floor_bw, 1e-3f);
    const float normalized = (density - frame->floor_bw) / range;
    // invert sign for the log-exposure axis; v_star is already baked into
    // frame->pivot_bw, so no separate "+ v_star" here
    const float x_adj = -(normalized - frame->pivot_bw);
    float v = frame->slope_bw * x_adj;
    v = _apply_snap_grade_density(v, d, frame->v_star, d->midtone_gamma, 0.0f, 0.0f);
    const float D_print = _hd_curve(v, frame->a_sh_bw, frame->a_hl_bw, frame->d_min_eff_bw, frame->d_max_eff_bw);
    const float linear = _apply_bpc(_density_to_linear(D_print), frame->bpc_black_bw);
    const float out = frame->anchor_scale * d->exposure * linear;
    for_each_channel(c) pix_out[c] = out;
  }
  else
  {
    // C41 (negative, invert sign) or E6 (slide, do not invert sign)
    const float invert_sign = (mode == DT_FILMINVERSION_E6) ? 1.0f : -1.0f;

    dt_aligned_pixel_t density;
    for_each_channel(c)
      density[c] = -log10f(fmaxf(pix_in[c], DENSITY_EPS));

    // the automatic per-channel normalization IS the mask neutralization;
    // mask_offset/filtration are optional manual trims on top
    dt_aligned_pixel_t normalized;
    for_each_channel(c)
    {
      const float range = fmaxf(frame->ceil[c] - frame->floor[c], 1e-3f);
      normalized[c] = (density[c] - frame->floor[c]) / range;
      normalized[c] = normalized[c] - d->mask_offset[c] + frame->filtration_eff[c];
    }

    dt_aligned_pixel_t x_adj;
    for_each_channel(c)
      x_adj[c] = invert_sign * (normalized[c] - frame->pivot[c]);

    // frame->slope[c] already includes k_trim and frame->pivot[c] already
    // bakes in v_star; frame->curv[c] is 0 unless Cast Removal is active.
    dt_aligned_pixel_t D_print;
    for_each_channel(c)
    {
      float v = frame->slope[c] * x_adj[c] + frame->curv[c] * normalized[c] * normalized[c];
      v = _apply_snap_grade_density(v, d, frame->v_star, d->midtone_gamma + d->snap_trim[c],
                                    d->shadow_filtration[c], d->highlight_filtration[c]);
      D_print[c] = _hd_curve(v, frame->a_sh[c], frame->a_hl[c], frame->d_min_eff[c], frame->d_max_eff[c]);
    }

    // Dye Separation + Separation Damping: density-domain saturation on the
    // print curve output. k=1 means identity, so skip when every channel's
    // slider (global + trim) is off.
    if(d->dye_k[0] > 1.0f || d->dye_k[1] > 1.0f || d->dye_k[2] > 1.0f)
      _apply_dye_separation(D_print, frame->d_min_eff, d->dye_k, d->separation_damping);

    for_each_channel(c)
    {
      const float linear = _apply_bpc(_density_to_linear(D_print[c]), frame->bpc_black[c]);
      pix_out[c] = frame->anchor_scale * d->exposure * linear;
    }
  }

  pix_out[3] = pix_in[3];
}

// Whole-frame measurement. Requires a host-readable width*height*4 buffer.
static void _measure(const float *const restrict in, const int width, const int height,
                     dt_iop_filminversion_measurement_t *const m)
{
  float *grid[3];
  const size_t ncells = _build_density_grid(in, width, height, grid);
  _percentile_floor_ceiling(grid, ncells, m->floor, m->ceil);

  // floor_ceil_range and textural_range are two different measurements of
  // this frame's density spread; damping their ratio is what sets the
  // slope-scaling range, not either measurement directly.
  const float floor_ceil_range = _floor_ceil_range(m->floor, m->ceil);
  const float textural_range = _measure_textural_range(grid, ncells);
  const float effective_range = _effective_grade_range(floor_ceil_range, textural_range);
  m->density_range = fmaxf(HD_TEXTURAL_RANGE_MIN, fminf(HD_TEXTURAL_RANGE_MAX, fabsf(effective_range)));

  m->anchor = _measure_anchor(grid, ncells, m->floor, m->ceil);

  // measured even when Cast Removal is off (cheap), so toggling it on an
  // already-cached frame doesn't force a re-measurement
  m->na = _measure_neutral_axis(grid, ncells, m->floor, m->ceil);
  _measure_shadow_tie_ref(grid, ncells, m->shadow_tie);

  for(int c = 0; c < 3; c++) dt_free_align(grid[c]);
}

// TRUE while the focused module causes the full uncropped scan -- film
// rebate, holder and all -- to be handed to us instead of the user's crop.
// Measuring that would read the orange base as if it were picture content.
//
// Two unrelated mechanisms do this, and both have to be caught:
//  - retouch, liquify and rotate/perspective filter the cropping modules out
//    of the pipe entirely, which shows up in operation_tags_filter()
//  - crop and clipping stay in the pipe but commit an identity crop while
//    they have focus, which only shows up in their own operation_tags()
// Both are gated on a basic pipe and on the module groups being active,
// matching the dt_iop_has_focus()/_skip_piece_on_tags() conditions.
static gboolean _cropping_filtered_out(const dt_dev_pixelpipe_iop_t *const piece)
{
  const dt_iop_module_t *const gui_module = dt_dev_gui_module();
  if(!gui_module
     || !dt_pipe_is_basic(piece->pipe)
     || !dt_dev_modulegroups_test_activated(darktable.develop))
    return FALSE;

  return (gui_module->operation_tags_filter() & IOP_TAG_CROPPING)
      || (gui_module->operation_tags() & IOP_TAG_CROPPING);
}

// Resolves this pipe's measurement, preferring the shared cache.
//
// Only the preview pipe always sees the whole image: the full pipe's roi
// shrinks to the viewport when zoomed past fit, so measuring from its own
// buffer would make mask neutralization drift with zoom/pan. The preview
// pipe therefore owns the cache, and the other pipes reuse even a stale
// entry in preference to re-measuring a cropped viewport.
//
// While the crop is filtered out of the pipe no measurement is trustworthy,
// so an existing entry is reused whatever its key and nothing is written
// back -- otherwise an uncropped measurement would be baked in and outlive
// the module that caused it.
//
// `in` may be NULL when the caller holds no host-readable buffer (the
// OpenCL path); FALSE is returned if a measurement would have been needed.
static gboolean _resolve_measurement(dt_iop_module_t *const self,
                                     dt_dev_pixelpipe_iop_t *const piece,
                                     const float *const restrict in,
                                     const int width, const int height,
                                     dt_iop_filminversion_measurement_t *const m)
{
  dt_iop_filminversion_gui_data_t *const g
      = (self->dev->gui_attached && self->gui_data) ? (dt_iop_filminversion_gui_data_t *)self->gui_data : NULL;
  const gboolean is_preview = dt_pipe_is_preview(piece->pipe);
  const gboolean uncropped = _cropping_filtered_out(piece);
  const dt_imgid_t imgid = self->dev->image_storage.id;

  // Hash of everything upstream of this module. Deliberately roi-less: that
  // drops both the roi and pipe->type from it, so it stays stable across
  // zoom/pan and matches between the preview and full pipes.
  const dt_hash_t hash = dt_dev_pixelpipe_piece_hash(piece, NULL, FALSE);

  gboolean have = FALSE;
  if(g && g->measure_cache_valid && g->measure_imgid == imgid)
  {
    const gboolean fresh = g->measure_hash == hash
                        && g->measure_width == width
                        && g->measure_height == height;
    if(uncropped || !is_preview || fresh)
    {
      *m = g->measurement;
      have = TRUE;
    }
  }

  if(!have)
  {
    if(!in) return FALSE;
    _measure(in, width, height, m);
    if(g && is_preview && !uncropped)
    {
      g->measurement = *m;
      g->measure_hash = hash;
      g->measure_width = width;
      g->measure_height = height;
      g->measure_imgid = imgid;
      g->measure_cache_valid = TRUE;
    }
  }

  // for the GUI-thread auto-mask-neutralize picker
  if(g)
    for(int c = 0; c < 4; c++) { g->last_floor[c] = m->floor[c]; g->last_ceil[c] = m->ceil[c]; }

  return TRUE;
}

// Per-channel curve parameters for one frame. Pure param math on an already
// resolved measurement -- touches no pixel data, so the OpenCL path builds
// this without needing any host buffer.
static dt_iop_filminversion_frame_t _frame_from_measurement(
    const dt_iop_filminversion_measurement_t *const m,
    const dt_iop_filminversion_data_t *const d)
{
  const float base_slope = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, d->grade_slope_factor * m->density_range));

  // Pivot/slope domain bridge. v_star, d_min_eff/d_max_eff and the zone
  // centers are in absolute log-density units; base_slope bridges those back
  // from the 0-1 normalized domain.
  //
  // invert_sign must be folded into the v_star/density terms here. This
  // module measures floor/ceil identically for both modes and inverts when
  // building x_adj (see _process_pixel), so applying NegPy's pivot formula
  // verbatim would double-apply the inversion for C41 -- v at the reference
  // tone would land on -v_star, washing the midtones out to paper white.
  const float invert_sign = (d->mode == DT_FILMINVERSION_E6) ? 1.0f : -1.0f;

  dt_iop_filminversion_frame_t frame;
  for(int c = 0; c < 4; c++) { frame.floor[c] = m->floor[c]; frame.ceil[c] = m->ceil[c]; frame.curv[c] = 0.0f; }
  for(int c = 0; c < 3; c++)
    frame.slope[c] = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, base_slope * d->k_trim[c]));

  // Grade-coupled toe/shoulder needs green's final per-frame slope (NegPy
  // couples to slopes[1] specifically, not an average) -- solved here now
  // that frame.slope[] is known, before it's needed for pivot below.
  _compute_curve_shape(d, frame.slope[1], frame.a_sh, frame.a_hl,
                       frame.d_min_eff, frame.d_max_eff, frame.bpc_black,
                       &frame.a_sh_bw, &frame.a_hl_bw, &frame.d_min_eff_bw, &frame.d_max_eff_bw,
                       &frame.bpc_black_bw, &frame.v_star, &frame.anchor_scale);

  for(int c = 0; c < 3; c++)
  {
    frame.pivot[c] = m->anchor
                   - invert_sign * (frame.v_star / frame.slope[c] + d->density * HD_DENSITY_MULTIPLIER);
    const float range = fmaxf(m->ceil[c] - m->floor[c], 1e-3f);
    frame.filtration_eff[c] = d->filtration[c] * HD_CMY_MAX_DENSITY / range;
  }

  frame.floor_bw = 0.2126f * m->floor[0] + 0.7152f * m->floor[1] + 0.0722f * m->floor[2];
  frame.ceil_bw = 0.2126f * m->ceil[0] + 0.7152f * m->ceil[1] + 0.0722f * m->ceil[2];
  frame.slope_bw = base_slope;
  frame.pivot_bw = m->anchor
                 - invert_sign * (frame.v_star / frame.slope_bw + d->density * HD_DENSITY_MULTIPLIER);

  // C41 only: the neutral-axis concept doesn't apply to B&W's single channel
  // or E6's already-positive relationship.
  if(d->mode == DT_FILMINVERSION_C41 && d->cast_removal_strength != 0.0f)
  {
    if(m->na.valid)
      _solve_cast_removal(d, &m->na, m->na.confidence, invert_sign, m->floor, m->ceil, &frame);
    else
      _solve_cast_removal_fallback(d, m->shadow_tie, invert_sign, m->anchor, m->floor, m->ceil, &frame);
  }

  return frame;
}

void process(dt_iop_module_t *const self, dt_dev_pixelpipe_iop_t *const piece,
             const void *const restrict ivoid, void *const restrict ovoid,
             const dt_iop_roi_t *const restrict roi_in, const dt_iop_roi_t *const restrict roi_out)
{
  const dt_iop_filminversion_data_t *const d = piece->data;
  assert(piece->colors == 4);

  const float *const restrict in = (float *)ivoid;
  float *const restrict out = (float *)ovoid;

  dt_iop_filminversion_measurement_t m;
  _resolve_measurement(self, piece, in, roi_in->width, roi_in->height, &m);
  const dt_iop_filminversion_frame_t frame = _frame_from_measurement(&m, d);

  DT_OMP_FOR()
  for(size_t k = 0; k < (size_t)roi_out->height * roi_out->width * 4; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;
    _process_pixel(pix_in, pix_out, d, &frame);
  }
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_filminversion_data_t *const d = piece->data;
  const dt_iop_filminversion_global_data_t *const gd = self->global_data;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  // The measurement is the only part that needs host pixels, so try the
  // shared cache before touching the device buffer. In the darkroom that
  // hits on every call once the preview pipe has run, which keeps the
  // interactive path completely free of device->host transfers.
  dt_iop_filminversion_measurement_t m;
  if(!_resolve_measurement(self, piece, NULL, width, height, &m))
  {
    // Cold cache: read this frame back once and measure it on the CPU,
    // exactly as process() would have.
    float *const buf = dt_alloc_align_float((size_t)width * height * 4);
    if(!buf) return DT_OPENCL_SYSMEM_ALLOCATION;

    const cl_int err
        = dt_opencl_copy_image_to_host(devid, buf, dev_in, width, height, sizeof(float) * 4);
    if(err != CL_SUCCESS)
    {
      dt_free_align(buf);
      return err;
    }

    _resolve_measurement(self, piece, buf, width, height, &m);
    dt_free_align(buf);
  }

  const dt_iop_filminversion_frame_t frame = _frame_from_measurement(&m, d);

  const int mode = d->mode;
  const float frame_bw[4] = { frame.floor_bw, frame.ceil_bw, frame.slope_bw, frame.pivot_bw };
  // a_sh/a_hl/d_min_eff/d_max_eff/bpc_black are per-channel (frame.a_sh etc,
  // packed below as their own float4 args); curve_bw carries the shared/
  // scalar version B&W uses instead.
  const float curve_bw[4] = { frame.a_sh_bw, frame.a_hl_bw, frame.d_min_eff_bw, frame.d_max_eff_bw };
  const float tone[4] = { frame.v_star, d->midtone_gamma, d->shadow_grade_gain, d->highlight_grade_gain };
  const float zone[4] = { d->shadow_density, d->highlight_density, frame.anchor_scale, d->exposure };
  // Snap's per-channel trim, pre-combined with the global value -- C41/E6
  // uses this; B&W (no per-channel trims) uses tone.y (d->midtone_gamma) instead.
  const float midtone_gamma_c[4] = { d->midtone_gamma + d->snap_trim[0], d->midtone_gamma + d->snap_trim[1],
                                     d->midtone_gamma + d->snap_trim[2], 0.0f };

  return dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_filminversion, width, height,
      CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(mode),
      CLFLARRAY(4, frame.floor), CLFLARRAY(4, frame.ceil),
      CLFLARRAY(4, frame.slope), CLFLARRAY(4, frame.pivot),
      CLFLARRAY(4, frame.curv), CLFLARRAY(4, frame.filtration_eff),
      CLFLARRAY(4, d->mask_offset), CLFLARRAY(4, frame_bw),
      CLFLARRAY(4, curve_bw),
      CLFLARRAY(4, frame.a_sh), CLFLARRAY(4, frame.a_hl),
      CLFLARRAY(4, frame.d_min_eff), CLFLARRAY(4, frame.d_max_eff),
      CLFLARRAY(4, frame.bpc_black), CLARG(frame.bpc_black_bw),
      CLFLARRAY(4, tone), CLFLARRAY(4, midtone_gamma_c), CLFLARRAY(4, zone),
      CLFLARRAY(4, d->shadow_filtration), CLFLARRAY(4, d->highlight_filtration),
      CLFLARRAY(4, d->dye_k), CLARG(d->separation_damping));
}
#endif

void init_global(dt_iop_module_so_t *self)
{
  const int program = 42; // filminversion.cl, from programs.conf
  dt_iop_filminversion_global_data_t *gd = malloc(sizeof(dt_iop_filminversion_global_data_t));
  self->data = gd;
  gd->kernel_filminversion = dt_opencl_create_kernel(program, "filminversion");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_filminversion_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_filminversion);
  free(self->data);
  self->data = NULL;
}

void init_presets(dt_iop_module_so_t *self)
{
  dt_iop_filminversion_params_t tmp = (dt_iop_filminversion_params_t){
    .mode = DT_FILMINVERSION_C41,
    .mask_offset = { 0.0f, 0.0f, 0.0f, 0.0f },
    .filtration = { 0.0f, 0.0f, 0.0f, 0.0f },
    .cast_removal_strength = 0.5f,
    .density = 0.5f,
    .grade = 127.5f,
    .toe = 0.0f,
    .shoulder = 0.0f,
    .toe_width = 1.0f,
    .shoulder_width = 1.0f,
    .exposure = 1.0f,
    .dye_separation = 0.0f,
    .separation_damping = 0.0f
  };

  dt_gui_presets_add_generic(_("color negative (C-41)"), self->op,
                             self->version(), &tmp, sizeof(tmp), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

  dt_iop_filminversion_params_t tmq = tmp;
  tmq.mode = DT_FILMINVERSION_BW;

  dt_gui_presets_add_generic(_("black and white negative"), self->op,
                             self->version(), &tmq, sizeof(tmq), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

  dt_iop_filminversion_params_t tmr = tmp;
  tmr.mode = DT_FILMINVERSION_E6;

  dt_gui_presets_add_generic(_("slide / reversal (E-6)"), self->op,
                             self->version(), &tmr, sizeof(tmr), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);
}

void init_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = g_malloc0(sizeof(dt_iop_filminversion_data_t));
}

void cleanup_pipe(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  g_free(piece->data);
  piece->data = NULL;
}


/* Global GUI stuff */

static void toggle_mode_controls(dt_iop_module_t *const self)
{
  dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  const dt_iop_filminversion_params_t *const p = self->params;

  const gboolean is_bw = (p->mode == DT_FILMINVERSION_BW);
  const gboolean is_c41 = (p->mode == DT_FILMINVERSION_C41);

  // meaningless for B&W: no orange mask
  gtk_widget_set_visible(g->mask_offset_R, !is_bw);
  gtk_widget_set_visible(g->mask_offset_G, !is_bw);
  gtk_widget_set_visible(g->mask_offset_B, !is_bw);
  gtk_widget_set_visible(g->auto_mask_button, !is_bw);
  gtk_widget_set_visible(g->filtration_R, !is_bw);
  gtk_widget_set_visible(g->filtration_G, !is_bw);
  gtk_widget_set_visible(g->filtration_B, !is_bw);
  gtk_widget_set_visible(g->temperature, !is_bw);
  gtk_widget_set_visible(g->shadow_temperature, !is_bw);
  gtk_widget_set_visible(g->highlight_temperature, !is_bw);
  gtk_widget_set_visible(g->cast_removal_strength, is_c41);

  // Dye separation is a colour operation and has no meaning in B&W.
  gtk_widget_set_visible(g->dye_separation, !is_bw);
  gtk_widget_set_visible(g->separation_damping, !is_bw);

  // Regional CMY is per-channel colour, meaningless for B&W.
  gtk_widget_set_visible(g->filtration_channel_box, !is_bw);
  gtk_widget_set_visible(g->shadow_filtration, !is_bw);
  gtk_widget_set_visible(g->highlight_filtration, !is_bw);

  // Per-channel trims are B&W-meaningless for the same reason k_trim etc are:
  // B&W has already collapsed to a single luminance channel by the time any
  // of these would apply.
  gtk_widget_set_visible(g->trim_channel_box, !is_bw);
  gtk_widget_set_visible(g->k_trim, !is_bw);
  gtk_widget_set_visible(g->snap_trim, !is_bw);
  gtk_widget_set_visible(g->toe_trim, !is_bw);
  gtk_widget_set_visible(g->toe_width_trim, !is_bw);
  gtk_widget_set_visible(g->shoulder_trim, !is_bw);
  gtk_widget_set_visible(g->shoulder_width_trim, !is_bw);
  gtk_widget_set_visible(g->dye_separation_trim_R, !is_bw);
  gtk_widget_set_visible(g->dye_separation_trim_G, !is_bw);
  gtk_widget_set_visible(g->dye_separation_trim_B, !is_bw);
}

// Optional manual trim on top of the automatic normalization: the user
// boxes a patch that should end up neutral, and this computes the
// mask_offset making that patch's normalized value equal across channels,
// via the latest cached floor/ceiling so it lands in the curve's domain.
// darktable re-invokes color_picker_apply() as the box is redrawn, so this
// updates live.
//
// Uses the patch MEAN, not a max/highlight statistic: a max-biased estimate
// only holds if the box contains nothing but uniform film base, and breaks
// down on a textured gray subject where each channel's max can come from a
// different bright spot.
static void apply_auto_mask_neutralize(dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_filminversion_gui_data_t *g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;

  dt_aligned_pixel_t normalized;
  for(int c = 0; c < 3; c++)
  {
    const float base = self->picked_color[c];
    const float density = -log10f(fmaxf(base, DENSITY_EPS));
    const float range = fmaxf(g->last_ceil[c] - g->last_floor[c], 1e-3f);
    normalized[c] = (density - g->last_floor[c]) / range;
  }

  // relative to green, so at least one channel stays at 0
  const float ref = normalized[1];
  for(int c = 0; c < 3; c++) p->mask_offset[c] = normalized[c] - ref;
  p->mask_offset[3] = 0.0f;

  ++darktable.gui->reset;
  dt_bauhaus_slider_set(g->mask_offset_R, p->mask_offset[0]);
  dt_bauhaus_slider_set(g->mask_offset_G, p->mask_offset[1]);
  dt_bauhaus_slider_set(g->mask_offset_B, p->mask_offset[2]);
  --darktable.gui->reset;

  dt_control_queue_redraw_widget(self->widget);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  if(darktable.gui->reset) return;
  dt_iop_filminversion_gui_data_t *g = self->gui_data;

  if(picker == g->auto_mask_button)
    apply_auto_mask_neutralize(self);
  else
    dt_print(DT_DEBUG_ALWAYS, "[filminversion] unknown color picker");
}

// Refreshes the 6 shared per-channel trim sliders to show g->trim_channel's
// stored values. dt_bauhaus_slider_set() DOES re-emit "value-changed" on a
// genuine value change (see _update_temperature_sliders() for where that
// bit -- this file's earlier assumption otherwise -- actually mattered).
// channelmixer.c's analogous output_callback gets away without a guard
// only because its wiring is one-directional (selector -> refresh display
// sliders, nothing wired back to the selector); this file has real cycles
// (channel-selector <-> shared slider, filtration <-> temperature), so
// guard unconditionally rather than rely on values happening to match.
static void _update_trim_sliders(dt_iop_module_t *self)
{
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  const dt_iop_filminversion_params_t *const p = self->params;
  const int c = g->trim_channel;
  // dt_bauhaus_slider_set() only skips "value-changed" when the new value
  // equals what the widget already shows -- it DOES re-emit on a genuine
  // change (contrary to what earlier comments in this file assumed). Any
  // caller that reaches this function from a context where the refreshed
  // value can legitimately differ from the widget's current display (e.g.
  // a sibling callback, not just gui_update() right after a fresh load) can
  // re-enter one of these sliders' own callbacks. Bracket unconditionally.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(g->k_trim, p->k_trim[c]);
  dt_bauhaus_slider_set(g->snap_trim, p->snap_trim[c]);
  dt_bauhaus_slider_set(g->toe_trim, p->toe_trim[c]);
  dt_bauhaus_slider_set(g->toe_width_trim, p->toe_width_trim[c]);
  dt_bauhaus_slider_set(g->shoulder_trim, p->shoulder_trim[c]);
  dt_bauhaus_slider_set(g->shoulder_width_trim, p->shoulder_width_trim[c]);
  DT_LEAVE_GUI_UPDATE();
}

// One shared callback for all three channel-select buttons: keeps exactly
// one active (radio-button behavior via plain toggle buttons -- same idiom
// as e.g. spots.c's shape-type buttons), updates g->trim_channel, and
// refreshes the 5 shared sliders to that channel's values.
static void _trim_channel_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  // gtk_toggle_button_set_active() re-emits "toggled" on a genuine state
  // change (same as dt_bauhaus_slider_set() re-emitting "value-changed" --
  // see _update_temperature_sliders()) -- so every set_active(..., FALSE)
  // call below for the two siblings re-enters this same callback
  // recursively (each of which flips another sibling, and so on) unless
  // guarded. DT_GUARD_GUI_UPDATE() alone doesn't help here: it
  // only guards against the *framework's* bulk gui_update() calls, not
  // against this function re-triggering itself. DT_ENTER_GUI_UPDATE()/
  // DT_LEAVE_GUI_UPDATE() bracket the state changes so nested re-entrant
  // calls hit the guard below and return immediately -- without this, a
  // single click recurses without bound (observed as a stack overflow).
  DT_GUARD_GUI_UPDATE();
  dt_iop_filminversion_gui_data_t *g = self->gui_data;

  DT_ENTER_GUI_UPDATE();

  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
  {
    // one channel must always stay selected -- undo an attempt to switch it off
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), TRUE);
    DT_LEAVE_GUI_UPDATE();
    return;
  }

  if(widget != g->trim_channel_R) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->trim_channel_R), FALSE);
  if(widget != g->trim_channel_G) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->trim_channel_G), FALSE);
  if(widget != g->trim_channel_B) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->trim_channel_B), FALSE);

  g->trim_channel = (widget == g->trim_channel_R) ? 0 : (widget == g->trim_channel_G) ? 1 : 2;

  DT_LEAVE_GUI_UPDATE();

  _update_trim_sliders(self);
}

// The 6 shared per-channel trim sliders write into p-><field>[g->trim_channel]
// instead of a fixed index -- can't use dt_bauhaus_slider_from_params()'s
// auto-binding for that, so each gets its own callback (same pattern as
// channelmixer.c's red_callback/green_callback/blue_callback).
static void _k_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->k_trim[g->trim_channel])
  {
    p->k_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _snap_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->snap_trim[g->trim_channel])
  {
    p->snap_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _toe_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->toe_trim[g->trim_channel])
  {
    p->toe_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _toe_width_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->toe_width_trim[g->trim_channel])
  {
    p->toe_width_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _shoulder_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->shoulder_trim[g->trim_channel])
  {
    p->shoulder_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _shoulder_width_trim_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->shoulder_width_trim[g->trim_channel])
  {
    p->shoulder_width_trim[g->trim_channel] = value;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

// Regional color: same shared-slider-plus-channel-selector idiom as the
// per-channel trims above, for shadow_filtration/highlight_filtration.
static void _update_filtration_sliders(dt_iop_module_t *self)
{
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  const dt_iop_filminversion_params_t *const p = self->params;
  const int c = g->filtration_channel;
  // see _update_trim_sliders(): dt_bauhaus_slider_set() re-emits on a
  // genuine value change, so bracket unconditionally.
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(g->shadow_filtration, p->shadow_filtration[c]);
  dt_bauhaus_slider_set(g->highlight_filtration, p->highlight_filtration[c]);
  DT_LEAVE_GUI_UPDATE();
}

static void _filtration_channel_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_filminversion_gui_data_t *g = self->gui_data;

  DT_ENTER_GUI_UPDATE();

  if(!gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
  {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget), TRUE);
    DT_LEAVE_GUI_UPDATE();
    return;
  }

  if(widget != g->filtration_channel_R) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->filtration_channel_R), FALSE);
  if(widget != g->filtration_channel_G) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->filtration_channel_G), FALSE);
  if(widget != g->filtration_channel_B) gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->filtration_channel_B), FALSE);

  g->filtration_channel = (widget == g->filtration_channel_R) ? 0 : (widget == g->filtration_channel_G) ? 1 : 2;

  DT_LEAVE_GUI_UPDATE();

  _update_filtration_sliders(self);
}

// Refreshes the temperature, shadow_temperature and highlight_temperature
// sliders to the Kelvin value implied by filtration/shadow_filtration/
// highlight_filtration's current green+blue pair. Called on gui_update, and
// after any direct edit of those M/Y sliders, so the Temperature lever's
// readout never goes stale relative to the values it actually operates on.
static void _update_temperature_sliders(dt_iop_module_t *self)
{
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  const dt_iop_filminversion_params_t *const p = self->params;
  // Reached from _shadow_filtration_callback/_highlight_filtration_callback
  // (a sibling slider's own callback, where the refreshed Kelvin value can
  // legitimately differ from what these widgets currently show), not just
  // gui_update() -- see _update_trim_sliders(). Without this guard, a
  // genuine change here can re-enter _temperature_callback and, from there,
  // filtration_G/B's own from_params handler: unbounded recursion, observed
  // as a stack overflow while dragging highlights filtration (blue channel).
  DT_ENTER_GUI_UPDATE();
  dt_bauhaus_slider_set(g->temperature, _wb_to_kelvin(p->filtration[1], p->filtration[2]));
  dt_bauhaus_slider_set(g->shadow_temperature, _wb_to_kelvin(p->shadow_filtration[1], p->shadow_filtration[2]));
  dt_bauhaus_slider_set(g->highlight_temperature, _wb_to_kelvin(p->highlight_filtration[1], p->highlight_filtration[2]));
  DT_LEAVE_GUI_UPDATE();
}

static void _shadow_filtration_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->shadow_filtration[g->filtration_channel])
  {
    p->shadow_filtration[g->filtration_channel] = value;
    _update_temperature_sliders(self);
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _highlight_filtration_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float value = dt_bauhaus_slider_get(slider);
  if(value != p->highlight_filtration[g->filtration_channel])
  {
    p->highlight_filtration[g->filtration_channel] = value;
    _update_temperature_sliders(self);
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

// filtration_G/filtration_B are dt_bauhaus_slider_from_params()-bound (own
// internal value-changed handler); this is an additional listener that only
// keeps the Temperature lever's readout in sync, no params/history touched.
static void _refresh_temperature_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  _update_temperature_sliders(self);
}

static void _temperature_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  const dt_iop_filminversion_gui_data_t *const g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  const float kelvin = dt_bauhaus_slider_get(slider);
  float m, y;
  _kelvin_to_wb(kelvin, p->filtration[1], p->filtration[2], &m, &y);
  if(m != p->filtration[1] || y != p->filtration[2])
  {
    p->filtration[1] = m;
    p->filtration[2] = y;
    // filtration_G/B are dt_bauhaus_slider_from_params()-bound, unlike the
    // manually-created sliders elsewhere in this file: setting one here
    // re-enters _refresh_temperature_callback synchronously, which would set
    // g->temperature right back -- while we're still inside *its* own
    // "value-changed" handler (stack overflow, observed on real testing).
    // Bracketing so the reentrant call hits DT_GUARD_GUI_UPDATE() and bails,
    // same fix as _trim_channel_callback/_filtration_channel_callback.
    DT_ENTER_GUI_UPDATE();
    dt_bauhaus_slider_set(g->filtration_G, m);
    dt_bauhaus_slider_set(g->filtration_B, y);
    DT_LEAVE_GUI_UPDATE();
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _shadow_temperature_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_filminversion_params_t *p = self->params;
  const float kelvin = dt_bauhaus_slider_get(slider);
  float m, y;
  _kelvin_to_wb(kelvin, p->shadow_filtration[1], p->shadow_filtration[2], &m, &y);
  if(m != p->shadow_filtration[1] || y != p->shadow_filtration[2])
  {
    p->shadow_filtration[1] = m;
    p->shadow_filtration[2] = y;
    _update_filtration_sliders(self);
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

static void _highlight_temperature_callback(GtkWidget *slider, dt_iop_module_t *self)
{
  DT_GUARD_GUI_UPDATE();
  dt_iop_filminversion_params_t *p = self->params;
  const float kelvin = dt_bauhaus_slider_get(slider);
  float m, y;
  _kelvin_to_wb(kelvin, p->highlight_filtration[1], p->highlight_filtration[2], &m, &y);
  if(m != p->highlight_filtration[1] || y != p->highlight_filtration[2])
  {
    p->highlight_filtration[1] = m;
    p->highlight_filtration[2] = y;
    _update_filtration_sliders(self);
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_filminversion_gui_data_t *g = IOP_GUI_ALLOC(filminversion);

  // defaults until the preview pipe caches a real measurement, so the picker
  // can't hit a degenerate 0/0 range if used immediately
  for(int c = 0; c < 4; c++) { g->last_floor[c] = 0.0f; g->last_ceil[c] = 1.0f; }

  static dt_action_def_t notebook_def = { };
  g->notebook = dt_ui_notebook_new(&notebook_def);
  dt_action_define_iop(self, NULL, N_("page"), GTK_WIDGET(g->notebook), &notebook_def);

  // Page MASK (mask neutralization + creative filtration)
  GtkWidget *page1 = self->widget = dt_ui_notebook_page(g->notebook, N_("mask"), NULL);

  dt_gui_box_add(page1, dt_ui_section_label_new(C_("section", "mask neutralization")));

  g->auto_mask_button = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, NULL);
  gtk_widget_set_tooltip_text(g->auto_mask_button, _("mask neutralization is fully automatic (whole-frame\n"
                                                      "per-channel measurement, no interaction needed).\n"
                                                      "this picker is only an optional manual trim: pick a\n"
                                                      "patch that should be neutral to nudge the mask trim\n"
                                                      "sliders below. re-analyzes live as the box is redrawn."));
  dt_action_define_iop(self, N_("pickers"), N_("mask neutralize"), g->auto_mask_button, &dt_action_def_toggle);
  dt_gui_box_add(page1, g->auto_mask_button);

  g->mask_offset_R = dt_bauhaus_slider_from_params(self, "mask_offset[0]");
  dt_bauhaus_widget_set_label(g->mask_offset_R, NULL, N_("mask trim red"));
  gtk_widget_set_tooltip_text(g->mask_offset_R, _("optional manual per-channel trim applied on top of the\n"
                                                   "automatic per-channel mask normalization (section 3B)."));

  g->mask_offset_G = dt_bauhaus_slider_from_params(self, "mask_offset[1]");
  dt_bauhaus_widget_set_label(g->mask_offset_G, NULL, N_("mask trim green"));
  gtk_widget_set_tooltip_text(g->mask_offset_G, _("optional manual per-channel trim applied on top of the\n"
                                                   "automatic per-channel mask normalization (section 3B)."));

  g->mask_offset_B = dt_bauhaus_slider_from_params(self, "mask_offset[2]");
  dt_bauhaus_widget_set_label(g->mask_offset_B, NULL, N_("mask trim blue"));
  gtk_widget_set_tooltip_text(g->mask_offset_B, _("optional manual per-channel trim applied on top of the\n"
                                                   "automatic per-channel mask normalization (section 3B)."));

  dt_gui_box_add(page1, dt_ui_section_label_new(C_("section", "color temperature")));

  g->temperature = dt_bauhaus_slider_new_with_range(self, HD_TEMP_MIN_KELVIN, HD_TEMP_MAX_KELVIN, 0, HD_TEMP_REF_KELVIN, 0);
  dt_bauhaus_widget_set_label(g->temperature, NULL, N_("temperature"));
  dt_bauhaus_slider_set_default(g->temperature, HD_TEMP_REF_KELVIN);
  gtk_widget_set_tooltip_text(g->temperature, _("kelvin lever over the filtration green/blue (magenta/\n"
                                                 "yellow) pair below -- an alternate, more intuitive way to\n"
                                                 "dial in the same two sliders. red/cyan is never touched\n"
                                                 "(real darkroom: cyan stays 0). nominal, not colorimetric."));
  g_signal_connect(G_OBJECT(g->temperature), "value-changed", G_CALLBACK(_temperature_callback), self);

  g->shadow_temperature = dt_bauhaus_slider_new_with_range(self, HD_TEMP_MIN_KELVIN, HD_TEMP_MAX_KELVIN, 0, HD_TEMP_REF_KELVIN, 0);
  dt_bauhaus_widget_set_label(g->shadow_temperature, NULL, N_("shadows temperature"));
  dt_bauhaus_slider_set_default(g->shadow_temperature, HD_TEMP_REF_KELVIN);
  gtk_widget_set_tooltip_text(g->shadow_temperature, _("kelvin lever over the shadows filtration green/blue\n"
                                                        "(magenta/yellow) pair (tone tab, regional color panel).\n"
                                                        "red/cyan is never touched (real darkroom: cyan stays 0).\n"
                                                        "nominal, not colorimetric."));
  g_signal_connect(G_OBJECT(g->shadow_temperature), "value-changed", G_CALLBACK(_shadow_temperature_callback), self);

  g->highlight_temperature = dt_bauhaus_slider_new_with_range(self, HD_TEMP_MIN_KELVIN, HD_TEMP_MAX_KELVIN, 0, HD_TEMP_REF_KELVIN, 0);
  dt_bauhaus_widget_set_label(g->highlight_temperature, NULL, N_("highlights temperature"));
  dt_bauhaus_slider_set_default(g->highlight_temperature, HD_TEMP_REF_KELVIN);
  gtk_widget_set_tooltip_text(g->highlight_temperature, _("kelvin lever over the highlights filtration green/blue\n"
                                                           "(magenta/yellow) pair (tone tab, regional color panel).\n"
                                                           "red/cyan is never touched (real darkroom: cyan stays 0).\n"
                                                           "nominal, not colorimetric."));
  g_signal_connect(G_OBJECT(g->highlight_temperature), "value-changed", G_CALLBACK(_highlight_temperature_callback), self);

  dt_gui_box_add(page1, g->temperature, g->shadow_temperature, g->highlight_temperature);

  dt_gui_box_add(page1, dt_ui_section_label_new(C_("section", "creative filtration")));

  g->filtration_R = dt_bauhaus_slider_from_params(self, "filtration[0]");
  dt_bauhaus_widget_set_label(g->filtration_R, NULL, N_("filtration red"));
  gtk_widget_set_tooltip_text(g->filtration_R, _("additive trim applied on top of the automatic mask\n"
                                                  "normalization, like an enlarger dichroic-head color filter.\n"
                                                  "scaled by the measured per-channel dynamic range, so a\n"
                                                  "slider unit stays gentle and consistent across scans."));

  g->filtration_G = dt_bauhaus_slider_from_params(self, "filtration[1]");
  dt_bauhaus_widget_set_label(g->filtration_G, NULL, N_("filtration green"));
  gtk_widget_set_tooltip_text(g->filtration_G, _("additive trim applied on top of the automatic mask\n"
                                                  "normalization, like an enlarger dichroic-head color filter.\n"
                                                  "scaled by the measured per-channel dynamic range, so a\n"
                                                  "slider unit stays gentle and consistent across scans."));

  g->filtration_B = dt_bauhaus_slider_from_params(self, "filtration[2]");
  dt_bauhaus_widget_set_label(g->filtration_B, NULL, N_("filtration blue"));
  gtk_widget_set_tooltip_text(g->filtration_B, _("additive trim applied on top of the automatic mask\n"
                                                  "normalization, like an enlarger dichroic-head color filter.\n"
                                                  "scaled by the measured per-channel dynamic range, so a\n"
                                                  "slider unit stays gentle and consistent across scans."));
  // extra listeners, on top of dt_bauhaus_slider_from_params()'s own binding,
  // that only keep the Temperature lever above in sync -- no params/history.
  g_signal_connect(G_OBJECT(g->filtration_G), "value-changed", G_CALLBACK(_refresh_temperature_callback), self);
  g_signal_connect(G_OBJECT(g->filtration_B), "value-changed", G_CALLBACK(_refresh_temperature_callback), self);

  dt_gui_box_add(page1, dt_ui_section_label_new(C_("section", "cast removal")));

  g->cast_removal_strength = dt_bauhaus_slider_from_params(self, "cast_removal_strength");
  dt_bauhaus_slider_set_digits(g->cast_removal_strength, 2);
  gtk_widget_set_tooltip_text(g->cast_removal_strength, _("fully automatic color-cast removal (C-41 only), zero\n"
                                                           "interaction needed: detects near-neutral reference\n"
                                                           "pixels in the shadow/mid/highlight tones and fits\n"
                                                           "red/blue to green's curve at those points.\n"
                                                           "0.5 (default) is usually neutral; lower drifts toward\n"
                                                           "a cyan cast, higher toward magenta/yellow, as the fit\n"
                                                           "under- or over-shoots the true correction amount."));

  // Page TONE (density/exposure + H&D curve)
  GtkWidget *page2 = self->widget = dt_ui_notebook_page(g->notebook, N_("tone"), NULL);

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "exposure")));

  // No auto-density picker: the automatic normalization and pivot/slope
  // bridge already anchor a well-exposed midtone without interaction.
  g->density = dt_bauhaus_slider_from_params(self, "density");
  dt_bauhaus_slider_set_feedback(g->density, 0);
  gtk_widget_set_tooltip_text(g->density, _("global density/exposure shift applied before the curve.\n"
                                            "higher values simulate more paper exposure (darker print)."));

  g->shadow_density = dt_bauhaus_slider_from_params(self, "shadow_density");
  dt_bauhaus_widget_set_label(g->shadow_density, NULL, N_("shadows density"));
  gtk_widget_set_tooltip_text(g->shadow_density, _("neutral brightness offset localized to the shadows only,\n"
                                                    "independent of the global density shift above."));

  g->highlight_density = dt_bauhaus_slider_from_params(self, "highlight_density");
  dt_bauhaus_widget_set_label(g->highlight_density, NULL, N_("highlights density"));
  gtk_widget_set_tooltip_text(g->highlight_density, _("neutral brightness offset localized to the highlights only,\n"
                                                       "independent of the global density shift above."));

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "regional color")));

  g->filtration_channel_R = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_RED, NULL);
  g->filtration_channel_G = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_GREEN, NULL);
  g->filtration_channel_B = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_BLUE, NULL);
  gtk_widget_set_tooltip_text(g->filtration_channel_R, _("edit the red channel's regional color tint below"));
  gtk_widget_set_tooltip_text(g->filtration_channel_G, _("edit the green channel's regional color tint below"));
  gtk_widget_set_tooltip_text(g->filtration_channel_B, _("edit the blue channel's regional color tint below"));
  g->filtration_channel = 0;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->filtration_channel_R), TRUE);
  g_signal_connect(G_OBJECT(g->filtration_channel_R), "toggled", G_CALLBACK(_filtration_channel_callback), self);
  g_signal_connect(G_OBJECT(g->filtration_channel_G), "toggled", G_CALLBACK(_filtration_channel_callback), self);
  g_signal_connect(G_OBJECT(g->filtration_channel_B), "toggled", G_CALLBACK(_filtration_channel_callback), self);
  g->filtration_channel_box = dt_gui_hbox(g->filtration_channel_R, g->filtration_channel_G, g->filtration_channel_B);
  gtk_widget_set_halign(g->filtration_channel_box, GTK_ALIGN_CENTER);
  dt_gui_box_add(page2, g->filtration_channel_box);

  g->shadow_filtration = dt_bauhaus_slider_new_with_range(self, -1.0f, 1.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->shadow_filtration, NULL, N_("shadows filtration"));
  dt_bauhaus_slider_set_default(g->shadow_filtration, 0.0f);
  gtk_widget_set_tooltip_text(g->shadow_filtration, _("creative per-channel color tint localized to the shadows\n"
                                                      "only, for whichever channel is selected above -- distinct\n"
                                                      "from split grade (contrast) and shadows density (neutral\n"
                                                      "brightness) above."));
  g_signal_connect(G_OBJECT(g->shadow_filtration), "value-changed", G_CALLBACK(_shadow_filtration_callback), self);

  g->highlight_filtration = dt_bauhaus_slider_new_with_range(self, -1.0f, 1.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->highlight_filtration, NULL, N_("highlights filtration"));
  dt_bauhaus_slider_set_default(g->highlight_filtration, 0.0f);
  gtk_widget_set_tooltip_text(g->highlight_filtration, _("creative per-channel color tint localized to the\n"
                                                         "highlights only, for whichever channel is selected above --\n"
                                                         "distinct from split grade (contrast) and highlights\n"
                                                         "density (neutral brightness) above."));
  g_signal_connect(G_OBJECT(g->highlight_filtration), "value-changed", G_CALLBACK(_highlight_filtration_callback), self);

  dt_gui_box_add(page2, g->shadow_filtration, g->highlight_filtration);

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "H&D curve")));

  g->grade = dt_bauhaus_slider_from_params(self, "grade");
  dt_bauhaus_slider_set_digits(g->grade, 0);
  dt_bauhaus_slider_set_feedback(g->grade, 0);
  gtk_widget_set_tooltip_text(g->grade, _("contrast on the ISO-R paper-grade scale.\n"
                                          "105 = softer (lower contrast), 150 = harder (higher contrast).\n"
                                          "range is narrowed to where this typically has a visible effect --\n"
                                          "beyond it, contrast is pinned to the paper's softest allowed slope."));

  g->shadow_grade = dt_bauhaus_slider_from_params(self, "shadow_grade");
  dt_bauhaus_slider_set_digits(g->shadow_grade, 0);
  gtk_widget_set_tooltip_text(g->shadow_grade, _("local contrast trim (ISO-R points) applied only in the shadows,\n"
                                                  "like a real darkroom split-grade exposure. negative = harder."));

  g->highlight_grade = dt_bauhaus_slider_from_params(self, "highlight_grade");
  dt_bauhaus_slider_set_digits(g->highlight_grade, 0);
  gtk_widget_set_tooltip_text(g->highlight_grade, _("local contrast trim (ISO-R points) applied only in the highlights,\n"
                                                     "like a real darkroom split-grade exposure. negative = harder."));

  g->snap = dt_bauhaus_slider_from_params(self, "snap");
  dt_bauhaus_slider_set_digits(g->snap, 2);
  gtk_widget_set_tooltip_text(g->snap, _("extra local contrast at the midtone centre (paper's variable-gamma\n"
                                         "S-curve). positive snaps contrast into the midtones, negative flattens it."));

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "toe / shoulder")));

  g->toe = dt_bauhaus_slider_from_params(self, "toe");
  dt_bauhaus_slider_set_digits(g->toe, 2);
  gtk_widget_set_tooltip_text(g->toe, _("shadow roll-off height.\n"
                                        "positive: lifts paper black (fogged shadows, real paper toe behavior).\n"
                                        "negative: sharpens the shadow knee instead."));

  g->shoulder = dt_bauhaus_slider_from_params(self, "shoulder");
  dt_bauhaus_slider_set_digits(g->shoulder, 2);
  gtk_widget_set_tooltip_text(g->shoulder, _("highlight roll-off height.\n"
                                             "positive: lifts paper white (compresses highlights toward grey).\n"
                                             "negative: sharpens the highlight knee instead."));

  g->toe_width = dt_bauhaus_slider_from_params(self, "toe_width");
  dt_bauhaus_slider_set_digits(g->toe_width, 2);
  gtk_widget_set_tooltip_text(g->toe_width, _("shadow knee sharpness.\n"
                                              "larger values make the toe roll-off more gradual."));

  g->shoulder_width = dt_bauhaus_slider_from_params(self, "shoulder_width");
  dt_bauhaus_slider_set_digits(g->shoulder_width, 2);
  gtk_widget_set_tooltip_text(g->shoulder_width, _("highlight knee sharpness.\n"
                                                   "larger values make the shoulder roll-off more gradual."));

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "per-channel trims")));

  // Round R/G/B channel-select buttons (dtgtk_cairo_paint_label, the same
  // paint function color-label thumbnail tags use), acting as a 3-way radio
  // group -- click one to choose which channel the 5 sliders below edit.
  // dtgtk_cairo_paint_label wants the dt_colorlabels_enum value directly in
  // its low bits (flags & 7) -- CPF_LABEL_RED/GREEN/BLUE are a *different*,
  // unrelated bitmask family (aliased to CPF_DIRECTION_* for the multi-dot
  // "flower" icon/thumbnail bitfield), not usable here despite the name.
  g->trim_channel_R = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_RED, NULL);
  g->trim_channel_G = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_GREEN, NULL);
  g->trim_channel_B = dtgtk_togglebutton_new(dtgtk_cairo_paint_label, DT_COLORLABELS_BLUE, NULL);
  gtk_widget_set_tooltip_text(g->trim_channel_R, _("edit the red channel's per-channel trims below"));
  gtk_widget_set_tooltip_text(g->trim_channel_G, _("edit the green channel's per-channel trims below"));
  gtk_widget_set_tooltip_text(g->trim_channel_B, _("edit the blue channel's per-channel trims below"));
  g->trim_channel = 0;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->trim_channel_R), TRUE);
  g_signal_connect(G_OBJECT(g->trim_channel_R), "toggled", G_CALLBACK(_trim_channel_callback), self);
  g_signal_connect(G_OBJECT(g->trim_channel_G), "toggled", G_CALLBACK(_trim_channel_callback), self);
  g_signal_connect(G_OBJECT(g->trim_channel_B), "toggled", G_CALLBACK(_trim_channel_callback), self);
  g->trim_channel_box = dt_gui_hbox(g->trim_channel_R, g->trim_channel_G, g->trim_channel_B);
  gtk_widget_set_halign(g->trim_channel_box, GTK_ALIGN_CENTER);
  dt_gui_box_add(page2, g->trim_channel_box);

#define FILMINVERSION_TRIM_TOOLTIP _("per-channel trim added on top of the global slider above.\n" \
                                     "corrects a crossover cast this specific control leaves behind --\n" \
                                     "a shared curve plus a constant offset can only neutralize one\n" \
                                     "tonal point, not the whole curve at once.")

  g->k_trim = dt_bauhaus_slider_new_with_range(self, 0.5f, 2.0f, 0, 1.0f, 2);
  dt_bauhaus_widget_set_label(g->k_trim, NULL, N_("contrast trim"));
  dt_bauhaus_slider_set_default(g->k_trim, 1.0f);
  gtk_widget_set_tooltip_text(g->k_trim, _("per-channel curve slope (contrast) trim, 1.0 = no change.\n"
                                           "corrects color casts that differ between shadows/mids/highlights\n"
                                           "(crossover), which mask offset alone cannot fix."));
  g_signal_connect(G_OBJECT(g->k_trim), "value-changed", G_CALLBACK(_k_trim_callback), self);

  g->snap_trim = dt_bauhaus_slider_new_with_range(self, -0.5f, 0.5f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->snap_trim, NULL, N_("snap trim"));
  dt_bauhaus_slider_set_default(g->snap_trim, 0.0f);
  gtk_widget_set_tooltip_text(g->snap_trim, FILMINVERSION_TRIM_TOOLTIP);
  g_signal_connect(G_OBJECT(g->snap_trim), "value-changed", G_CALLBACK(_snap_trim_callback), self);

  g->toe_trim = dt_bauhaus_slider_new_with_range(self, -1.0f, 1.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->toe_trim, NULL, N_("toe trim"));
  dt_bauhaus_slider_set_default(g->toe_trim, 0.0f);
  gtk_widget_set_tooltip_text(g->toe_trim, FILMINVERSION_TRIM_TOOLTIP);
  g_signal_connect(G_OBJECT(g->toe_trim), "value-changed", G_CALLBACK(_toe_trim_callback), self);

  g->toe_width_trim = dt_bauhaus_slider_new_with_range(self, -2.0f, 2.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->toe_width_trim, NULL, N_("toe width"));
  dt_bauhaus_slider_set_default(g->toe_width_trim, 0.0f);
  gtk_widget_set_tooltip_text(g->toe_width_trim, FILMINVERSION_TRIM_TOOLTIP);
  g_signal_connect(G_OBJECT(g->toe_width_trim), "value-changed", G_CALLBACK(_toe_width_trim_callback), self);

  g->shoulder_trim = dt_bauhaus_slider_new_with_range(self, -1.0f, 1.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->shoulder_trim, NULL, N_("shoulder trim"));
  dt_bauhaus_slider_set_default(g->shoulder_trim, 0.0f);
  gtk_widget_set_tooltip_text(g->shoulder_trim, FILMINVERSION_TRIM_TOOLTIP);
  g_signal_connect(G_OBJECT(g->shoulder_trim), "value-changed", G_CALLBACK(_shoulder_trim_callback), self);

  g->shoulder_width_trim = dt_bauhaus_slider_new_with_range(self, -2.0f, 2.0f, 0, 0.0f, 2);
  dt_bauhaus_widget_set_label(g->shoulder_width_trim, NULL, N_("shoulder width"));
  dt_bauhaus_slider_set_default(g->shoulder_width_trim, 0.0f);
  gtk_widget_set_tooltip_text(g->shoulder_width_trim, FILMINVERSION_TRIM_TOOLTIP);
  g_signal_connect(G_OBJECT(g->shoulder_width_trim), "value-changed", G_CALLBACK(_shoulder_width_trim_callback), self);

  dt_gui_box_add(page2, g->k_trim, g->snap_trim, g->toe_trim, g->toe_width_trim, g->shoulder_trim, g->shoulder_width_trim);

  dt_gui_box_add(page2, dt_ui_section_label_new(C_("section", "output")));

  g->exposure = dt_bauhaus_slider_from_params(self, "exposure");
  dt_bauhaus_slider_set_soft_min(g->exposure, 0.25);
  dt_bauhaus_slider_set_soft_max(g->exposure, 4.0);
  dt_bauhaus_slider_set_digits(g->exposure, 2);
  gtk_widget_set_tooltip_text(g->exposure, _("anchor trim applied after the curve.\n"
                                             "fine-tunes overall brightness before tone mapping."));

  // Page COLOR (density-domain saturation / dye separation)
  GtkWidget *page3 = self->widget = dt_ui_notebook_page(g->notebook, N_("color"), NULL);

  dt_gui_box_add(page3, dt_ui_section_label_new(C_("section", "dye separation")));

  g->dye_separation = dt_bauhaus_slider_from_params(self, "dye_separation");
  dt_bauhaus_slider_set_digits(g->dye_separation, 2);
  gtk_widget_set_tooltip_text(g->dye_separation, _("density-domain saturation control applied after the H&D curve.\n"
                                                   "0 = identity, 1 = full separation: muted colour is pushed outward\n"
                                                   "and the matrix k*I+(1-k)*J redistributes density differences."));

  g->separation_damping = dt_bauhaus_slider_from_params(self, "separation_damping");
  dt_bauhaus_slider_set_digits(g->separation_damping, 2);
  gtk_widget_set_tooltip_text(g->separation_damping, _("per-pixel damping of the separation push by the pixel's own chroma.\n"
                                                       "0 = flat push for all pixels; 1 = vivid pixels are pulled back while\n"
                                                       "muted pixels keep the full push, preventing oversaturated colours."));

  dt_gui_box_add(page3, dt_ui_section_label_new(C_("section", "per-channel trim")));

  g->dye_separation_trim_R = dt_bauhaus_slider_from_params(self, "dye_separation_trim[0]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_R, NULL, N_("dye separation trim red"));
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_R, 2);
  gtk_widget_set_tooltip_text(g->dye_separation_trim_R, FILMINVERSION_TRIM_TOOLTIP);

  g->dye_separation_trim_G = dt_bauhaus_slider_from_params(self, "dye_separation_trim[1]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_G, NULL, N_("dye separation trim green"));
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_G, 2);
  gtk_widget_set_tooltip_text(g->dye_separation_trim_G, FILMINVERSION_TRIM_TOOLTIP);

  g->dye_separation_trim_B = dt_bauhaus_slider_from_params(self, "dye_separation_trim[2]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_B, NULL, N_("dye separation trim blue"));
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_B, 2);
  gtk_widget_set_tooltip_text(g->dye_separation_trim_B, FILMINVERSION_TRIM_TOOLTIP);

#undef FILMINVERSION_TRIM_TOOLTIP

  self->widget = dt_gui_vbox();

  g->mode = dt_bauhaus_combobox_from_params(self, "mode");
  gtk_widget_set_tooltip_text(g->mode, _("select the film type: color negative (C-41),\n"
                                         "black & white negative, or slide/reversal (E-6)."));

  dt_gui_box_add(self->widget, g->notebook);
}


void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_filminversion_gui_data_t *g = self->gui_data;

  if(!w || w == g->mode)
  {
    toggle_mode_controls(self);
  }
}


void gui_update(dt_iop_module_t *const self)
{
  dt_iop_color_picker_reset(self, TRUE);
  gui_changed(self, NULL, NULL);
  // g->trim_channel/g->filtration_channel are pure GUI state (not params)
  // and aren't reset here -- only the shared sliders' displayed values need
  // refreshing, in case the underlying params changed (undo/redo, preset,
  // image switch) while a non-default channel was selected.
  _update_trim_sliders(self);
  _update_filtration_sliders(self);
  _update_temperature_sliders(self);
}

void gui_reset(dt_iop_module_t *self)
{
  dt_iop_color_picker_reset(self, TRUE);
  dt_iop_filminversion_gui_data_t *g = self->gui_data;
  g->trim_channel = 0;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->trim_channel_R), TRUE);
  g->filtration_channel = 0;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->filtration_channel_R), TRUE);
}
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
