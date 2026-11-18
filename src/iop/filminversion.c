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
#include "common/color_picker.h"
#include "common/darktable.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/openmp_maths.h"
#include "develop/pixelpipe_hb.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "gui/color_picker_proxy.h"
#include "iop/iop_api.h"
#include "libs/colorpicker.h"
#include "libs/lib.h"

#include <glib.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Converts a linear scene-referred film-negative scan into a linear
 * scene-referred positive via a density-space H&D characteristic curve.
 *
 * The module deliberately does only what darktable has no other module for:
 * mask neutralization, plus the creative filtration and dye separation trims
 * that act in the film's own density domain. Exposure, contrast, tone and
 * color grading are left to the rest of the pipeline, so the print curve
 * runs at fixed, neutral settings.
 *
 * Sits after demosaic/white-balance, before exposure/filmic.
 *   INPUT:  linear scene-referred RGB, not inverted, not clipped to [0,1].
 *   DOMAIN: density = -log10(clamp(linear, eps)); all shaping happens here.
 *   OUTPUT: linear = 10^-density, rescaled so the curve pivot lands on 18%
 *           mid-gray. Deliberately NOT clipped -- downstream filmic/tone
 *           mapping owns highlight handling.
 *
 * Mask neutralization is a measurement, not a live analysis: the film's
 * per-channel density bounds (plus the slope, anchor and cast-removal
 * references derived from the same pass) are measured once and stored in
 * the params. A fresh instance measures the whole frame on its first preview
 * run; after that only the picker re-measures, over the area it is drawn on.
 * Nothing changes with crop, zoom or upstream edits until the user picks.
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

// the print curve's fixed settings, at what used to be the sliders' defaults
#define HD_GRADE 127.5f                // ISO-R paper grade
#define HD_DENSITY 0.5f                // density shift before the curve
#define HD_CAST_REMOVAL_STRENGTH 0.5f  // NegPy's own Cast Removal default

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

// Section 3B (per-channel floor/ceiling normalization): reduces the
// measured area to a coarse block-median grid, then measures three things
// from it -- the floor/ceiling normalization range (see _measure_bounds()),
// its luma-weighted spread (floor_ceil_range), and the tighter P10-P90
// "textural" range. The latter two feed HD_AUTO_GRADE_*'s damping formula,
// which is what sets the slope-scaling density_range, NOT either raw
// measurement directly.
#define HD_GRID_TARGET 1024        // NegPy analysis_grid
// Fraction of width/height trimmed off each edge of the whole-frame
// measurement (NegPy analysis_buffer default), so uncropped scan borders
// (rebate, sprocket holes) don't pollute the statistics. A picked area is
// taken as drawn, as NegPy does with its analysis_rect.
#define HD_ANALYSIS_EDGE_MARGIN 0.05f
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

// NegPy's bounds pass (normalization.py analyze_log_exposure_bounds_from_log
// and negpy_core bounds.rs same_pixel_color_floor_refs), see _measure_bounds().
// Percentiles, not fractions.
#define HD_BOUNDS_LUMA_CLIP 0.01         // NegPy base_luma_clip
#define HD_BOUNDS_COLOR_CLIP 1.0         // NegPy base_color_clip
#define HD_BOUNDS_BAND_WIDTH 4.0         // NegPy color_bounds_band_width
#define HD_BOUNDS_CHROMA_QUANTILE 0.30   // NegPy neutral_axis_chroma_quantile
#define HD_BOUNDS_CHROMA_CAP 0.29        // NegPy neutral_axis_chroma_cap
#define HD_BOUNDS_FIRST_PASS_CAP 0.55    // NegPy neutral_axis_first_pass_cap
#define HD_BOUNDS_MIN_CELLS 64           // NegPy neutral_axis_min_pixels
// NegPy's luma weights for the bounds pass only (negpy/domain/types.py); the
// rest of this file still weights by Rec.709
#define HD_BOUNDS_LUMA_R 0.2974
#define HD_BOUNDS_LUMA_G 0.6273
#define HD_BOUNDS_LUMA_B 0.0753
// NegPy's to_log_density clamps transmittance to [1e-6, 1]
#define HD_BOUNDS_MAX_DENSITY 6.0f

#define HD_TEXTURAL_LO_PERCENTILE 10.0f  // NegPy textural_range_clip=10.0
#define HD_TEXTURAL_HI_PERCENTILE 90.0f
#define HD_TEXTURAL_RANGE_MIN 0.3f
#define HD_TEXTURAL_RANGE_MAX 3.5f
// Deliberately lower than NegPy's slope_min=2.0: with HD_GRADE_CONTRAST_SCALE
// lowered from NegPy's 2.9 for a softer baseline look, base_slope sits right
// around 2.0 for a nominal image, and a 2.0 floor would pin it there.
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
// centred on, and how strongly the density shift moves it.
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

// toe/shoulder are scaled by this before use (the d_max_eff/d_min_eff lift)
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


DT_MODULE_INTROSPECTION(9, dt_iop_filminversion_params_t)


typedef enum dt_iop_filminversion_mode_t
{
  DT_FILMINVERSION_C41 = 0, // $DESCRIPTION: "color negative (C-41)"
  DT_FILMINVERSION_BW = 1,  // $DESCRIPTION: "black & white negative"
} dt_iop_filminversion_mode_t;


typedef struct dt_iop_filminversion_params_t
{
  dt_iop_filminversion_mode_t mode; /* $DEFAULT: DT_FILMINVERSION_C41 $DESCRIPTION: "film type" */

  /* per-channel density offset subtracted after the measured normalization,
     for what the measurement leaves behind */
  float mask_offset[4];  /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "mask trim" */

  /* creative filtration: per-channel additive density trim applied after
     mask neutralization (mimics enlarger dichroic-head CMY filtration,
     which is itself an additive density-unit shift). */
  float filtration[4];   /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "filtration" */

  /* density-domain saturation on the print curve output, per channel: the
     separation gain is 1 + trim, so 0 is identity */
  float dye_separation_trim[4]; /* $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "dye separation trim" */

  /* the stored mask-neutralization measurement, no widgets: written by the
     first preview run and by the picker, see _store_measurement(). FALSE
     makes the next preview run measure the whole frame. */
  gboolean measured;      /* $DEFAULT: FALSE */
  float floor[4];         // per-channel density floor
  float ceil[4];          // per-channel density ceiling
  float density_range;    // damped spread driving the slope scaling
  float anchor;           // auto exposure anchor
  float shadow_tie[4];    // Cast Removal one-point fallback reference
  gboolean axis_valid;    /* $DEFAULT: FALSE */
  gboolean axis_has_highlight; /* $DEFAULT: FALSE */
  float axis_confidence;
  float axis_mid_ref[4];
  float axis_shadow_ref[4];
  float axis_highlight_ref[4];
} dt_iop_filminversion_params_t;


// Cast Removal neutral-axis measurement result, see _measure_neutral_axis().
typedef struct dt_iop_filminversion_neutral_axis_t
{
  gboolean valid;                    // mid+shadow bands both found
  gboolean has_highlight;            // highlight band also found (enables curvature)
  float confidence;                  // 0..1, scales the cast removal strength
  dt_aligned_pixel_t mid_ref;        // per-channel raw log density at the midtone ref
  dt_aligned_pixel_t shadow_ref;     // per-channel raw log density at the shadow ref
  dt_aligned_pixel_t highlight_ref;  // per-channel raw log density at the highlight ref
} dt_iop_filminversion_neutral_axis_t;

// Everything _measure() derives from the input buffer. Depends only on the
// pixels, never on this module's params.
typedef struct dt_iop_filminversion_measurement_t
{
  dt_aligned_pixel_t floor;       // per-channel measured density floor
  dt_aligned_pixel_t ceil;        // per-channel measured density ceiling
  dt_aligned_pixel_t shadow_tie;  // Cast Removal one-point fallback reference
  float density_range;            // damped spread driving the slope scaling
  float anchor;                   // auto exposure anchor
  dt_iop_filminversion_neutral_axis_t na;
} dt_iop_filminversion_measurement_t;

typedef struct dt_iop_filminversion_data_t
{
  dt_iop_filminversion_mode_t mode;
  dt_aligned_pixel_t mask_offset;
  dt_aligned_pixel_t filtration;
  dt_aligned_pixel_t dye_k;    // per-channel separation gain, 1 + trim
  gboolean measured;
  dt_iop_filminversion_measurement_t m;
} dt_iop_filminversion_data_t;

// Per-frame quantities, recomputed once per process() call from the
// measurement. Separate from dt_iop_filminversion_data_t because these
// depend on the measurement, which an unmeasured instance takes from the
// pixels.
typedef struct dt_iop_filminversion_frame_t
{
  dt_aligned_pixel_t floor;          // per-channel measured density floor
  dt_aligned_pixel_t ceil;           // per-channel measured density ceiling
  dt_aligned_pixel_t slope;          // per-channel clamped slope (C41)
  dt_aligned_pixel_t pivot;          // per-channel pivot (NegPy compute_pivot)
  dt_aligned_pixel_t curv;           // per-channel Cast Removal curvature (0 unless active)
  dt_aligned_pixel_t filtration_eff; // per-channel HD_CMY_MAX_DENSITY-scaled filtration
  float floor_bw, ceil_bw;           // luma-combined equivalents for B&W mode
  float slope_bw, pivot_bw;

  // curve shape (see _compute_curve_shape()), shared by every channel
  float a_sh;            // toe knee sharpness
  float a_hl;            // shoulder knee sharpness
  float d_min_eff;       // shoulder-adjusted paper white
  float d_max_eff;       // toe-adjusted paper black
  float bpc_black;       // black point compensation clip
  float v_star;          // pivot value solved to hit HD_ANCHOR_TARGET_DENSITY
  float anchor_scale;    // normalizes the curve pivot to 18% mid-gray
} dt_iop_filminversion_frame_t;


typedef struct dt_iop_filminversion_gui_data_t
{
  GtkWidget *mode;
  GtkWidget *picker;
  GtkWidget *mask_offset_R, *mask_offset_G, *mask_offset_B;
  GtkWidget *filtration_R, *filtration_G, *filtration_B;
  GtkWidget *dye_separation_trim_R, *dye_separation_trim_G, *dye_separation_trim_B;

  // Measurements handed from the preview pipe to the GUI thread, which is
  // the only one allowed to write params. Guarded by the module's gui lock.
  //  - auto: the whole-frame pass of an unmeasured instance, committed by
  //    _preview_pipe_finished_callback(). Non-preview pipes render with it
  //    meanwhile, so a zoomed full pipe doesn't measure its viewport.
  //  - picked: the picker area, committed by color_picker_apply().
  gboolean auto_valid;
  dt_imgid_t auto_imgid;
  dt_iop_filminversion_measurement_t auto_m;
  gboolean picked_valid;
  dt_imgid_t picked_imgid;
  dt_iop_filminversion_measurement_t picked_m;
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

// No tiling: an unmeasured instance measures whole-frame statistics, so
// per-tile buffers would give each tile its own floor/ceiling and produce
// seams.
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
  // versions 5 to 8 carried the whole tone/color control set this module
  // no longer has. Only the controls that survived are carried over; the
  // rest fall back to the fixed print settings, and the measurement is
  // redone, since those versions never stored one.
  //
  // pre-5 was an unrelated power-curve prototype with no mapping at all.
  if(old_version > 8) return 1;

  dt_iop_filminversion_params_t *n = calloc(1, sizeof(dt_iop_filminversion_params_t));
  if(!n) return 1;
  memcpy(n, self->default_params, sizeof(dt_iop_filminversion_params_t));

  if(old_version >= 5)
  {
    // mode, mask_offset and filtration lead every one of these layouts
    typedef struct dt_iop_filminversion_params_v5_head_t
    {
      int mode;
      float mask_offset[4];
      float filtration[4];
    } dt_iop_filminversion_params_v5_head_t;

    const dt_iop_filminversion_params_v5_head_t *o = old_params;
    // E-6 (2) was dropped: its edits come back as a color negative
    n->mode = o->mode == DT_FILMINVERSION_BW ? DT_FILMINVERSION_BW : DT_FILMINVERSION_C41;
    for(int c = 0; c < 4; c++)
    {
      n->mask_offset[c] = o->mask_offset[c];
      n->filtration[c] = o->filtration[c];
    }
  }

  if(old_version == 8)
  {
    typedef struct dt_iop_filminversion_params_v8_t
    {
      int mode;
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
      float toe_trim[4];
      float shoulder_trim[4];
      float toe_width_trim[4];
      float shoulder_width_trim[4];
      float snap_trim[4];
      float dye_separation_trim[4];
    } dt_iop_filminversion_params_v8_t;

    const dt_iop_filminversion_params_v8_t *o = old_params;
    // the old gain was 1 + dye_separation + trim; folding the global slider
    // into each trim keeps the per-channel gain
    for(int c = 0; c < 4; c++)
      n->dye_separation_trim[c] = fmaxf(-1.0f, fminf(1.0f, o->dye_separation + o->dye_separation_trim[c]));
  }

  *new_params = n;
  *new_params_size = sizeof(dt_iop_filminversion_params_t);
  *new_version = 9;
  return 0;
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

// Density-domain saturation on the print curve's output, in place: k*I +
// (1-k)*J per channel around the pixel's mean density, so neutrals are
// preserved. k=1 is identity. The curve shape is shared by all channels, so
// the paper-white base NegPy subtracts first cancels out of this form.
static inline void _apply_dye_separation(float *const restrict D, const dt_aligned_pixel_t k)
{
  const float mean = (D[0] + D[1] + D[2]) / 3.0f;
  for(int c = 0; c < 3; c++) D[c] = mean + k[c] * (D[c] - mean);
}

static void _measurement_from_params(const dt_iop_filminversion_params_t *const p,
                                     dt_iop_filminversion_measurement_t *const m)
{
  for(int c = 0; c < 4; c++)
  {
    m->floor[c] = p->floor[c];
    m->ceil[c] = p->ceil[c];
    m->shadow_tie[c] = p->shadow_tie[c];
    m->na.mid_ref[c] = p->axis_mid_ref[c];
    m->na.shadow_ref[c] = p->axis_shadow_ref[c];
    m->na.highlight_ref[c] = p->axis_highlight_ref[c];
  }
  m->density_range = p->density_range;
  m->anchor = p->anchor;
  m->na.valid = p->axis_valid;
  m->na.has_highlight = p->axis_has_highlight;
  m->na.confidence = p->axis_confidence;
}

static void _store_measurement(dt_iop_filminversion_params_t *const p,
                               const dt_iop_filminversion_measurement_t *const m)
{
  for(int c = 0; c < 4; c++)
  {
    p->floor[c] = m->floor[c];
    p->ceil[c] = m->ceil[c];
    p->shadow_tie[c] = m->shadow_tie[c];
    p->axis_mid_ref[c] = m->na.mid_ref[c];
    p->axis_shadow_ref[c] = m->na.shadow_ref[c];
    p->axis_highlight_ref[c] = m->na.highlight_ref[c];
  }
  p->density_range = m->density_range;
  p->anchor = m->anchor;
  p->axis_valid = m->na.valid;
  p->axis_has_highlight = m->na.has_highlight;
  p->axis_confidence = m->na.confidence;
  p->measured = TRUE;
}

void commit_params(dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_filminversion_params_t *const p = (dt_iop_filminversion_params_t *)p1;
  dt_iop_filminversion_data_t *const d = piece->data;

  d->mode = p->mode;
  for(size_t c = 0; c < 4; c++) d->mask_offset[c] = p->mask_offset[c];
  for(size_t c = 0; c < 4; c++) d->filtration[c] = p->filtration[c];
  // NegPy per_channel_dye_separation's own [0,3] range
  for(size_t c = 0; c < 4; c++)
    d->dye_k[c] = fmaxf(0.0f, fminf(3.0f, 1.0f + p->dye_separation_trim[c]));

  d->measured = p->measured;
  if(p->measured) _measurement_from_params(p, &d->m);
}

// Toe/shoulder curve shape on the neutral paper: both start at zero, and
// only the grade coupling (NegPy grade_coupled_shape) lifts them, so the
// softer-knee branches NegPy uses for negative toe/shoulder never apply.
// v_star/anchor_scale come from the same shape.
static void _compute_curve_shape(const float base_slope, dt_iop_filminversion_frame_t *const frame)
{
  const float slope_norm = fmaxf(0.0f, fminf(1.0f, (base_slope - HD_SLOPE_MIN) / (HD_SLOPE_MAX - HD_SLOPE_MIN)));
  const float toe = HD_TOE_GRADE_STRENGTH * slope_norm * HD_TOE_SHOULDER_STRENGTH;
  const float shoulder = HD_SHOULDER_GRADE_STRENGTH * slope_norm * HD_TOE_SHOULDER_STRENGTH;

  // toe lifts the paper-black ceiling, shoulder the paper-white floor
  frame->d_max_eff = HD_D_MAX - toe * HD_TOE_HEIGHT;
  frame->a_sh = HD_TOE_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF;
  frame->d_min_eff = HD_D_MIN + shoulder * HD_SHOULDER_HEIGHT;
  frame->a_hl = HD_SHOULDER_SHARPNESS_BASE * HD_TOESHOULDER_WIDTH_REF;
  if(frame->d_max_eff < frame->d_min_eff + 0.1f) frame->d_max_eff = frame->d_min_eff + 0.1f;

  // Black point compensation clip point (NegPy bpc_black): references the
  // *physical* d_max, not d_max_eff, so the toe's lifted black still gets
  // remapped to true black.
  frame->bpc_black = _density_to_linear(HD_D_MAX);

  // Pivot solved onto HD_ANCHOR_TARGET_DENSITY, then that fixed density
  // anchored to 18% mid-gray -- so anchor_scale never drifts with grade.
  frame->v_star = _solve_v_star(frame->a_sh, frame->a_hl, frame->d_min_eff, frame->d_max_eff);
  frame->anchor_scale = 0.18f / _density_to_linear(HD_ANCHOR_TARGET_DENSITY);
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

// Block-median density grid shared by the bounds, textural-range and
// neutral-axis measurements, over [ax0,ax1) x [ay0,ay1) of a buffer `width`
// pixels wide. The per-block median of a small fixed sample is what makes
// the statistics robust to dust specks and grain. With reject_border, the
// unexposed surround is dropped before returning, so the count returned may
// be lower than the grid dimensions imply. Caller frees grid_out[0..2] via
// dt_free_align().
static size_t _build_density_grid(const float *const restrict in, const int width,
                                  const int ax0, const int ay0, const int ax1, const int ay1,
                                  const gboolean reject_border, float *grid_out[3])
{
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
  return reject_border ? _reject_unexposed_border(grid_out, gw, gh) : ncells;
}

static int _cmp_float(const void *a, const void *b)
{
  const float x = *(const float *)a, y = *(const float *)b;
  return (x > y) - (x < y);
}

static int _cmp_double(const void *a, const void *b)
{
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

// np.percentile's default linear interpolation, on an ascending array
static double _percentile_sorted_f(const float *const s, const size_t n, const double q)
{
  if(n == 0) return 0.0;
  const double idx = (double)(n - 1) * (q / 100.0);
  const size_t lo = MIN((size_t)fmax(floor(idx), 0.0), n - 1);
  const size_t hi = MIN(lo + 1, n - 1);
  return (double)s[lo] + (idx - (double)lo) * ((double)s[hi] - (double)s[lo]);
}

static double _percentile_sorted_d(const double *const s, const size_t n, const double q)
{
  if(n == 0) return 0.0;
  const double idx = (double)(n - 1) * (q / 100.0);
  const size_t lo = MIN((size_t)fmax(floor(idx), 0.0), n - 1);
  const size_t hi = MIN(lo + 1, n - 1);
  return s[lo] + (idx - (double)lo) * (s[hi] - s[lo]);
}

// sorts v in place
static inline double _percentile_d(double *const v, const size_t n, const double q)
{
  qsort(v, n, sizeof(double), _cmp_double);
  return _percentile_sorted_d(v, n, q);
}

static inline double _median3(const double v[3])
{
  return fmax(fmin(v[0], v[1]), fmin(fmax(v[0], v[1]), v[2]));
}

// The lowest-chroma HD_BOUNDS_CHROMA_QUANTILE of a band, with chroma read
// on each channel's offset divided by `gamma` (its provisional span). Fills
// sel with the selected row indices and returns their median chroma, or a
// negative value when too few rows survive. scratch holds n doubles.
static double _bounds_chroma_select(const double (*const d)[3], const size_t n, const double gamma[3],
                                    size_t *const sel, size_t *const nsel, double *const chroma,
                                    double *const scratch)
{
  double g[3];
  for(int c = 0; c < 3; c++) g[c] = fabs(gamma[c]) < 1e-6 ? 1e-6 : gamma[c];

  for(size_t i = 0; i < n; i++)
  {
    const double r = d[i][0] / g[0], gg = d[i][1] / g[1], b = d[i][2] / g[2];
    chroma[i] = sqrt(((r - gg) * (r - gg) + (gg - b) * (gg - b) + (r - b) * (r - b)) / 3.0);
  }
  memcpy(scratch, chroma, sizeof(double) * n);
  const double thr = _percentile_d(scratch, n, HD_BOUNDS_CHROMA_QUANTILE * 100.0);

  size_t k = 0;
  for(size_t i = 0; i < n; i++)
    if(chroma[i] <= thr) sel[k++] = i;
  *nsel = k;
  if(k < HD_BOUNDS_MIN_CELLS) return -1.0;

  for(size_t i = 0; i < k; i++) scratch[i] = chroma[sel[i]];
  return _percentile_d(scratch, k, 50.0);
}

static void _bounds_column_medians(const double (*const d)[3], const size_t *const sel, const size_t nsel,
                                   double *const scratch, double out[3])
{
  for(int c = 0; c < 3; c++)
  {
    for(size_t i = 0; i < nsel; i++) scratch[i] = d[sel[i]][c];
    out[c] = _percentile_d(scratch, nsel, 50.0);
  }
}

// NegPy same_pixel_color_floor_refs, in its log10-transmittance domain: the
// dense-end per-channel refs read off ONE shared set of grid cells, the
// lowest-chroma part of the luma-extreme band, instead of an independent
// percentile per channel. Separate percentiles land on a different scene
// object per channel, so colored highlight content passes for film cast; a
// shared chroma-gated set cannot. FALSE when that set is too small or too
// chromatic, and the caller keeps the percentile refs.
static gboolean _bounds_dense_refs(float *const L[3], const size_t n,
                                   const double luma_floors[3], const double luma_ceils[3],
                                   const double base[3], double out[3])
{
  double denom[3];
  for(int c = 0; c < 3; c++)
  {
    const double dd = luma_ceils[c] - luma_floors[c];
    denom[c] = fabs(dd) < 1e-6 ? (dd >= 0.0 ? 1e-6 : -1e-6) : dd;
  }

  float *const luma = malloc(sizeof(float) * n);
  float *const sorted = malloc(sizeof(float) * n);
  double (*const d)[3] = malloc(sizeof(double) * 3 * n);
  size_t *const sel = malloc(sizeof(size_t) * n);
  double *const chroma = malloc(sizeof(double) * n);
  double *const scratch = malloc(sizeof(double) * n);
  gboolean found = FALSE;
  if(!luma || !sorted || !d || !sel || !chroma || !scratch) goto done;

  for(size_t i = 0; i < n; i++)
    luma[i] = (float)(HD_BOUNDS_LUMA_R * (L[0][i] - luma_floors[0]) / denom[0]
                    + HD_BOUNDS_LUMA_G * (L[1][i] - luma_floors[1]) / denom[1]
                    + HD_BOUNDS_LUMA_B * (L[2][i] - luma_floors[2]) / denom[2]);
  memcpy(sorted, luma, sizeof(float) * n);
  qsort(sorted, n, sizeof(float), _cmp_float);
  const double clip = fmax(0.00001, fmin(50.0 - HD_BOUNDS_BAND_WIDTH, HD_BOUNDS_COLOR_CLIP));
  const double lo = _percentile_sorted_f(sorted, n, clip);
  const double hi = _percentile_sorted_f(sorted, n, clip + HD_BOUNDS_BAND_WIDTH);

  size_t nb = 0;
  for(size_t i = 0; i < n; i++)
  {
    if(luma[i] < lo || luma[i] > hi) continue;
    for(int c = 0; c < 3; c++) d[nb][c] = L[c][i] - base[c];
    nb++;
  }
  if(nb < HD_BOUNDS_MIN_CELLS) goto done;

  double spans[3];
  for(int c = 0; c < 3; c++) spans[c] = luma_floors[c] - base[c];
  size_t nsel = 0;
  const double first_chroma = _bounds_chroma_select((const double (*)[3])d, nb, spans, sel, &nsel, chroma, scratch);
  // a loose first cap: pass 2 would self-normalize a homogeneous colored
  // cluster to zero chroma and read it as neutral
  if(first_chroma < 0.0 || first_chroma > HD_BOUNDS_FIRST_PASS_CAP) goto done;

  double provisional[3];
  _bounds_column_medians((const double (*)[3])d, sel, nsel, scratch, provisional);
  if(fabs(provisional[0]) < 1e-6 || fabs(provisional[1]) < 1e-6 || fabs(provisional[2]) < 1e-6) goto done;

  const double second_chroma = _bounds_chroma_select((const double (*)[3])d, nb, provisional, sel, &nsel, chroma, scratch);
  if(second_chroma < 0.0 || second_chroma > HD_BOUNDS_CHROMA_CAP) goto done;

  double m[3];
  _bounds_column_medians((const double (*)[3])d, sel, nsel, scratch, m);
  for(int c = 0; c < 3; c++) out[c] = base[c] + m[c];
  found = TRUE;

done:
  free(scratch);
  free(chroma);
  free(sel);
  free(d);
  free(sorted);
  free(luma);
  return found;
}

// This IS the mask neutralization: the per-channel density window that
// absorbs the orange mask's offset. A port of NegPy's
// analyze_log_exposure_bounds_from_log at its default clips.
//
// Two passes are recombined. The luma pass (gentle HD_BOUNDS_LUMA_CLIP)
// sets how far the window reaches, i.e. the dynamic range. The color pass
// (HD_BOUNDS_COLOR_CLIP) sets each channel's position, i.e. the cast: its
// thin end from per-channel percentiles (density on film is bounded below
// by base, so those are trustworthy), its dense end from the shared
// chroma-gated set of _bounds_dense_refs().
//
// NegPy works in log10 transmittance, where the dense end is the low
// "floor"; ours is density, where the thin end is. Hence L = -D going in,
// and floor = -ceil, ceil = -floor coming out.
static void _measure_bounds(float *const grid[3], const size_t ncells,
                            dt_aligned_pixel_t floor_out, dt_aligned_pixel_t ceil_out)
{
  float *L[3];
  float *sorted[3];
  for(int c = 0; c < 3; c++)
  {
    L[c] = malloc(sizeof(float) * ncells);
    sorted[c] = malloc(sizeof(float) * ncells);
  }

  double lf[3], lc[3], cf[3], cc[3];
  for(int c = 0; c < 3; c++)
  {
    for(size_t i = 0; i < ncells; i++)
      L[c][i] = -fmaxf(0.0f, fminf(HD_BOUNDS_MAX_DENSITY, grid[c][i]));
    memcpy(sorted[c], L[c], sizeof(float) * ncells);
    qsort(sorted[c], ncells, sizeof(float), _cmp_float);
    lf[c] = _percentile_sorted_f(sorted[c], ncells, HD_BOUNDS_LUMA_CLIP);
    lc[c] = _percentile_sorted_f(sorted[c], ncells, 100.0 - HD_BOUNDS_LUMA_CLIP);
    cf[c] = _percentile_sorted_f(sorted[c], ncells, HD_BOUNDS_COLOR_CLIP);
    cc[c] = _percentile_sorted_f(sorted[c], ncells, 100.0 - HD_BOUNDS_COLOR_CLIP);
  }

  double sp[3];
  if(_bounds_dense_refs(L, ncells, lf, lc, cc, sp))
    for(int c = 0; c < 3; c++) cf[c] = sp[c];

  double floors[3], ceils[3];
  const double spans[3] = { cf[0] - cc[0], cf[1] - cc[1], cf[2] - cc[2] };
  const double total = spans[0] + spans[1] + spans[2];
  if(spans[0] * total > 0.0 && spans[1] * total > 0.0 && spans[2] * total > 0.0)
  {
    // the luma window reaches past the color refs by one shared fraction of
    // each channel's own span: a reach in log units would tint a film whose
    // channels differ in density range
    const double reach_f = ((lf[0] - cf[0]) + (lf[1] - cf[1]) + (lf[2] - cf[2])) / total;
    const double reach_c = ((lc[0] - cc[0]) + (lc[1] - cc[1]) + (lc[2] - cc[2])) / total;
    // offset by the base's mean-minus-median, the placement NegPy's print
    // constants are tuned to
    const double shift = (cc[0] + cc[1] + cc[2] - 3.0 * _median3(cc)) / total;
    for(int c = 0; c < 3; c++)
    {
      floors[c] = cf[c] + (reach_f + shift) * spans[c];
      ceils[c] = cc[c] + (reach_c + shift) * spans[c];
    }
  }
  else
  {
    const double mean_lf = (lf[0] + lf[1] + lf[2]) / 3.0, mean_lc = (lc[0] + lc[1] + lc[2]) / 3.0;
    const double mean_cf = _median3(cf), mean_cc = _median3(cc);
    for(int c = 0; c < 3; c++)
    {
      floors[c] = mean_lf + (cf[c] - mean_cf);
      ceils[c] = mean_lc + (cc[c] - mean_cc);
    }
  }

  for(int c = 0; c < 3; c++)
  {
    floor_out[c] = (float)-ceils[c];
    ceil_out[c] = (float)-floors[c];
    if(ceil_out[c] - floor_out[c] < 1e-3f) ceil_out[c] = floor_out[c] + 1e-3f;
    free(sorted[c]);
    free(L[c]);
  }
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
static void _solve_cast_removal(const dt_iop_filminversion_neutral_axis_t *const na,
                                float confidence, float invert_sign,
                                const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil,
                                dt_iop_filminversion_frame_t *const frame)
{
  const float strength = HD_CAST_REMOVAL_STRENGTH * confidence;
  // NegPy (per_channel_curve_params(), logic.py) only ever applies this
  // correction for strength > 0.
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
    // clamp_dev()).
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
    // difference). Left uncorrected, the solved slope comes out negative for
    // C41 and the always-positive [HD_SLOPE_MIN, HD_SLOPE_MAX] clamp below
    // floors it to the wrong value entirely. Re-multiplying by invert_sign
    // cancels it back out to a true, always-positive magnitude.
    float slope_ch = (fabsf(du) < 1e-6f) ? slope_g : invert_sign * ((t_m - t_s) - curv * (u_m * u_m - u_s * u_s)) / du;
    slope_ch = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, slope_ch));

    // pivot re-pin so the channel's curve passes exactly through (u_m, t_m):
    // slope_ch*invert*(u_m-pivot_ch) + curv*u_m^2 = t_m
    float pivot_ch = pivot_g;
    if(fabsf(slope_ch) > 1e-6f)
      pivot_ch = u_m - invert_sign * (t_m - curv * u_m * u_m) / slope_ch;

    frame->slope[ch] = slope_ch;
    frame->pivot[ch] = pivot_ch;
    frame->curv[ch] = curv;
  }
}

// One-point fallback: tilts each channel's slope so its shadow-side
// reference lands on green's, pinned through the pivot's reference tone.
// Used when the neutral-axis measurement above isn't available.
static void _solve_cast_removal_fallback(const dt_aligned_pixel_t shadow_tie_ref, float invert_sign,
                                         float anchor,
                                         const dt_aligned_pixel_t floor, const dt_aligned_pixel_t ceil,
                                         dt_iop_filminversion_frame_t *const frame)
{
  const float strength = HD_CAST_REMOVAL_STRENGTH;

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

    frame->slope[ch] = slope_ch;
    frame->pivot[ch] = anchor
                      - invert_sign * (frame->v_star / slope_ch + HD_DENSITY * HD_DENSITY_MULTIPLIER);
    frame->curv[ch] = 0.0f;
  }
}

// linear -> log10 density -> per-channel normalization -> mask/filtration
// trims -> pivot/slope bridge -> H&D curve -> dye separation -> linear ->
// mid-gray anchor.
static inline void _process_pixel(const dt_aligned_pixel_t pix_in,
                                  dt_aligned_pixel_t pix_out,
                                  const dt_iop_filminversion_data_t *const d,
                                  const dt_iop_filminversion_frame_t *const frame,
                                  const gboolean dye_separation)
{
  if(d->mode == DT_FILMINVERSION_BW)
  {
    // B&W: one luminance density, no mask neutralization or filtration.
    const float luma = fmaxf(0.2126f * pix_in[0] + 0.7152f * pix_in[1] + 0.0722f * pix_in[2], DENSITY_EPS);
    const float density = -log10f(luma);
    const float range = fmaxf(frame->ceil_bw - frame->floor_bw, 1e-3f);
    const float normalized = (density - frame->floor_bw) / range;
    // invert sign for the log-exposure axis; v_star is already baked into
    // frame->pivot_bw, so no separate "+ v_star" here
    const float v = frame->slope_bw * -(normalized - frame->pivot_bw);
    const float D_print = _hd_curve(v, frame->a_sh, frame->a_hl, frame->d_min_eff, frame->d_max_eff);
    const float out = frame->anchor_scale * _apply_bpc(_density_to_linear(D_print), frame->bpc_black);
    for_each_channel(c) pix_out[c] = out;
  }
  else
  {
    // the measured per-channel normalization IS the mask neutralization;
    // mask_offset/filtration are manual trims on top
    dt_aligned_pixel_t D_print;
    for_each_channel(c)
    {
      const float density = -log10f(fmaxf(pix_in[c], DENSITY_EPS));
      const float range = fmaxf(frame->ceil[c] - frame->floor[c], 1e-3f);
      const float normalized = (density - frame->floor[c]) / range - d->mask_offset[c] + frame->filtration_eff[c];
      // negative: invert the log-exposure axis. frame->pivot[c] already
      // bakes in v_star; frame->curv[c] is 0 unless Cast Removal found a
      // highlight reference.
      const float v = frame->slope[c] * -(normalized - frame->pivot[c]) + frame->curv[c] * normalized * normalized;
      D_print[c] = _hd_curve(v, frame->a_sh, frame->a_hl, frame->d_min_eff, frame->d_max_eff);
    }

    if(dye_separation) _apply_dye_separation(D_print, d->dye_k);

    for_each_channel(c)
      pix_out[c] = frame->anchor_scale * _apply_bpc(_density_to_linear(D_print[c]), frame->bpc_black);
  }

  pix_out[3] = pix_in[3];
}

// Measurement over box (x0, y0, x1, y1) of a width x height buffer, or over
// the whole frame minus HD_ANALYSIS_EDGE_MARGIN, with the unexposed
// surround rejected, when box is NULL. Requires a host-readable buffer.
static void _measure(const float *const restrict in, const int width, const int height,
                     const int *const box, dt_iop_filminversion_measurement_t *const m)
{
  int ax0, ay0, ax1, ay1;
  if(box)
  {
    ax0 = box[0]; ay0 = box[1]; ax1 = box[2]; ay1 = box[3];
  }
  else
  {
    const int mx = (int)(width * HD_ANALYSIS_EDGE_MARGIN);
    const int my = (int)(height * HD_ANALYSIS_EDGE_MARGIN);
    ax0 = mx;
    ay0 = my;
    ax1 = width - mx > ax0 ? width - mx : width;
    ay1 = height - my > ay0 ? height - my : height;
  }

  float *grid[3];
  const size_t ncells = _build_density_grid(in, width, ax0, ay0, ax1, ay1, box == NULL, grid);
  _measure_bounds(grid, ncells, m->floor, m->ceil);

  // floor_ceil_range and textural_range are two different measurements of
  // this frame's density spread; damping their ratio is what sets the
  // slope-scaling range, not either measurement directly.
  const float floor_ceil_range = _floor_ceil_range(m->floor, m->ceil);
  const float textural_range = _measure_textural_range(grid, ncells);
  const float effective_range = _effective_grade_range(floor_ceil_range, textural_range);
  m->density_range = fmaxf(HD_TEXTURAL_RANGE_MIN, fminf(HD_TEXTURAL_RANGE_MAX, fabsf(effective_range)));

  m->anchor = _measure_anchor(grid, ncells, m->floor, m->ceil);

  // also taken for B&W, so switching to C-41 needs no new measurement
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

// TRUE when this pipe run is the one darktable samples our picker from:
// the conditions of _request_color_pick() in pixelpipe_hb.c.
static gboolean _picker_active(const dt_iop_module_t *const self, const dt_dev_pixelpipe_iop_t *const piece)
{
  const dt_develop_t *const dev = self->dev;
  return dev->gui_attached
      && self->gui_data
      && piece->pipe == dev->preview_pipe
      && darktable.lib->proxy.colorpicker.picker_proxy
      && darktable.lib->proxy.colorpicker.primary_sample
      && self == dev->gui_module
      && self->enabled
      && self->request_color_pick != DT_REQUEST_COLORPICK_OFF;
}

// TRUE when _resolve_measurement() needs host pixels for this pipe run.
static gboolean _needs_pixels(dt_iop_module_t *const self, dt_dev_pixelpipe_iop_t *const piece)
{
  const dt_iop_filminversion_data_t *const d = piece->data;
  if(_picker_active(self, piece)) return TRUE;
  if(d->measured) return FALSE;

  dt_iop_filminversion_gui_data_t *const g = self->dev->gui_attached ? self->gui_data : NULL;
  if(!g || dt_pipe_is_preview(piece->pipe)) return TRUE;

  dt_iop_gui_enter_critical_section(self);
  const gboolean have = g->auto_valid && g->auto_imgid == self->dev->image_storage.id;
  dt_iop_gui_leave_critical_section(self);
  return !have;
}

// Resolves the measurement this pipe renders with, and hands fresh ones to
// the GUI thread for storing. `in` may be NULL only when _needs_pixels() is
// FALSE.
//
// A stored measurement always wins. An unmeasured instance measures the
// whole frame; only the preview pipe always sees all of it (the full pipe's
// roi shrinks to the viewport when zoomed past fit), so the preview pipe's
// measurement is the one committed, and the other pipes borrow it until
// then.
static void _resolve_measurement(dt_iop_module_t *const self,
                                 dt_dev_pixelpipe_iop_t *const piece,
                                 const float *const restrict in,
                                 const dt_iop_roi_t *const roi_in,
                                 dt_iop_filminversion_measurement_t *const m)
{
  const dt_iop_filminversion_data_t *const d = piece->data;
  dt_iop_filminversion_gui_data_t *const g = self->dev->gui_attached ? self->gui_data : NULL;
  const dt_imgid_t imgid = self->dev->image_storage.id;
  const int width = roi_in->width;
  const int height = roi_in->height;

  if(in && _picker_active(self, piece))
  {
    int box[4];
    if(!dt_color_picker_box(self, roi_in, darktable.lib->proxy.colorpicker.primary_sample,
                            PIXELPIPE_PICKER_INPUT, box))
    {
      dt_iop_filminversion_measurement_t pm;
      _measure(in, width, height, box, &pm);
      dt_iop_gui_enter_critical_section(self);
      g->picked_m = pm;
      g->picked_imgid = imgid;
      g->picked_valid = TRUE;
      dt_iop_gui_leave_critical_section(self);
    }
  }

  if(d->measured)
  {
    *m = d->m;
    return;
  }

  const gboolean is_preview = dt_pipe_is_preview(piece->pipe);
  if(g && !is_preview)
  {
    dt_iop_gui_enter_critical_section(self);
    const gboolean have = g->auto_valid && g->auto_imgid == imgid;
    if(have) *m = g->auto_m;
    dt_iop_gui_leave_critical_section(self);
    if(have) return;
  }

  _measure(in, width, height, NULL, m);

  // with the crop filtered out of the pipe, the frame we were handed
  // includes the rebate: render with it, but don't store it
  if(g && is_preview && !_cropping_filtered_out(piece))
  {
    dt_iop_gui_enter_critical_section(self);
    g->auto_m = *m;
    g->auto_imgid = imgid;
    g->auto_valid = TRUE;
    dt_iop_gui_leave_critical_section(self);
  }
}

// Per-channel curve parameters for one frame. Pure param math on an already
// resolved measurement -- touches no pixel data, so the OpenCL path builds
// this without needing any host buffer.
static dt_iop_filminversion_frame_t _frame_from_measurement(
    const dt_iop_filminversion_measurement_t *const m,
    const dt_iop_filminversion_data_t *const d)
{
  // The fixed grade sits on the ISO-R scale mirrored around [105,150]'s
  // midpoint, which is where 127.5 maps onto itself.
  const float grade_slope_factor = HD_GRADE_CONTRAST_SCALE / (HD_GRADE / 100.0f);
  const float base_slope = fmaxf(HD_SLOPE_MIN, fminf(HD_SLOPE_MAX, grade_slope_factor * m->density_range));

  // Pivot/slope domain bridge. v_star, d_min_eff/d_max_eff are in absolute
  // log-density units; base_slope bridges those back from the 0-1
  // normalized domain.
  //
  // invert_sign must be folded into the v_star/density terms here. This
  // module measures floor/ceil in density and inverts when building v (see
  // _process_pixel), so applying NegPy's pivot formula verbatim would
  // double-apply the inversion -- v at the reference tone would land on
  // -v_star, washing the midtones out to paper white.
  const float invert_sign = -1.0f;

  dt_iop_filminversion_frame_t frame;
  for(int c = 0; c < 4; c++)
  {
    frame.floor[c] = m->floor[c];
    frame.ceil[c] = m->ceil[c];
    frame.slope[c] = base_slope;
    frame.curv[c] = 0.0f;
  }

  // grade-coupled toe/shoulder needs green's per-frame slope (NegPy
  // couples to slopes[1]), which is base_slope before cast removal
  _compute_curve_shape(base_slope, &frame);

  for(int c = 0; c < 3; c++)
  {
    frame.pivot[c] = m->anchor
                   - invert_sign * (frame.v_star / frame.slope[c] + HD_DENSITY * HD_DENSITY_MULTIPLIER);
    const float range = fmaxf(m->ceil[c] - m->floor[c], 1e-3f);
    frame.filtration_eff[c] = d->filtration[c] * HD_CMY_MAX_DENSITY / range;
  }
  frame.pivot[3] = 0.0f;
  frame.filtration_eff[3] = 0.0f;

  frame.floor_bw = 0.2126f * m->floor[0] + 0.7152f * m->floor[1] + 0.0722f * m->floor[2];
  frame.ceil_bw = 0.2126f * m->ceil[0] + 0.7152f * m->ceil[1] + 0.0722f * m->ceil[2];
  frame.slope_bw = base_slope;
  frame.pivot_bw = m->anchor
                 - invert_sign * (frame.v_star / frame.slope_bw + HD_DENSITY * HD_DENSITY_MULTIPLIER);

  // C41 only: the neutral-axis concept doesn't apply to B&W's single channel
  if(d->mode == DT_FILMINVERSION_C41)
  {
    if(m->na.valid)
      _solve_cast_removal(&m->na, m->na.confidence, invert_sign, m->floor, m->ceil, &frame);
    else
      _solve_cast_removal_fallback(m->shadow_tie, invert_sign, m->anchor, m->floor, m->ceil, &frame);
  }

  return frame;
}

static inline gboolean _wants_dye_separation(const dt_iop_filminversion_data_t *const d)
{
  return d->dye_k[0] != 1.0f || d->dye_k[1] != 1.0f || d->dye_k[2] != 1.0f;
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
  _resolve_measurement(self, piece, in, roi_in, &m);
  const dt_iop_filminversion_frame_t frame = _frame_from_measurement(&m, d);
  const gboolean dye_separation = _wants_dye_separation(d);

  DT_OMP_FOR()
  for(size_t k = 0; k < (size_t)roi_out->height * roi_out->width * 4; k += 4)
  {
    const float *const restrict pix_in = in + k;
    float *const restrict pix_out = out + k;
    _process_pixel(pix_in, pix_out, d, &frame, dye_separation);
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

  // A stored measurement needs no host pixels, which keeps the usual path
  // free of device->host transfers. Measuring reads the frame back once.
  dt_iop_filminversion_measurement_t m;
  if(_needs_pixels(self, piece))
  {
    float *const buf = dt_alloc_align_float((size_t)width * height * 4);
    if(!buf) return DT_OPENCL_SYSMEM_ALLOCATION;

    const cl_int err
        = dt_opencl_copy_image_to_host(devid, buf, dev_in, width, height, sizeof(float) * 4);
    if(err != CL_SUCCESS)
    {
      dt_free_align(buf);
      return err;
    }

    _resolve_measurement(self, piece, buf, roi_in, &m);
    dt_free_align(buf);
  }
  else
    _resolve_measurement(self, piece, NULL, roi_in, &m);

  const dt_iop_filminversion_frame_t frame = _frame_from_measurement(&m, d);

  const int mode = d->mode;
  const int dye_separation = _wants_dye_separation(d);
  const float frame_bw[4] = { frame.floor_bw, frame.ceil_bw, frame.slope_bw, frame.pivot_bw };
  const float curve[4] = { frame.a_sh, frame.a_hl, frame.d_min_eff, frame.d_max_eff };

  return dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_filminversion, width, height,
      CLARG(dev_in), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(mode),
      CLFLARRAY(4, frame.floor), CLFLARRAY(4, frame.ceil),
      CLFLARRAY(4, frame.slope), CLFLARRAY(4, frame.pivot),
      CLFLARRAY(4, frame.curv), CLFLARRAY(4, frame.filtration_eff),
      CLFLARRAY(4, d->mask_offset), CLFLARRAY(4, frame_bw), CLFLARRAY(4, curve),
      CLARG(frame.bpc_black), CLARG(frame.anchor_scale),
      CLFLARRAY(4, d->dye_k), CLARG(dye_separation));
}
#endif

void init_global(dt_iop_module_so_t *self)
{
  const int program = 200; // filminversion.cl, from programs.conf
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
  // unmeasured, so applying either one measures the frame it lands on
  dt_iop_filminversion_params_t tmp = (dt_iop_filminversion_params_t){ .mode = DT_FILMINVERSION_C41 };

  dt_gui_presets_add_generic(_("color negative (C-41)"), self->op,
                             self->version(), &tmp, sizeof(tmp), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

  tmp.mode = DT_FILMINVERSION_BW;

  dt_gui_presets_add_generic(_("black and white negative"), self->op,
                             self->version(), &tmp, sizeof(tmp), TRUE, DEVELOP_BLEND_CS_RGB_SCENE);
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

// Stores the measurement of the area the picker is drawn on. darktable
// calls this after each preview run that moved the picker, which is also
// the run _resolve_measurement() measured the area on.
void color_picker_apply(dt_iop_module_t *self, GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  if(darktable.gui->reset) return;
  dt_iop_filminversion_gui_data_t *g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;

  dt_iop_filminversion_measurement_t m;
  dt_iop_gui_enter_critical_section(self);
  const gboolean have = g->picked_valid && g->picked_imgid == self->dev->image_storage.id;
  if(have) m = g->picked_m;
  g->picked_valid = FALSE;
  dt_iop_gui_leave_critical_section(self);
  if(!have) return;

  _store_measurement(p, &m);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

// Stores the whole-frame measurement of an unmeasured instance, once.
static void _preview_pipe_finished_callback(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_filminversion_gui_data_t *g = self->gui_data;
  dt_iop_filminversion_params_t *p = self->params;
  dt_develop_t *dev = self->dev;
  if(!g || p->measured || !self->enabled) return;

  // storing adds a history item, which would discard everything above an
  // earlier history step the user is looking at
  if(dev->history_end != g_list_length(dev->history)) return;

  dt_iop_filminversion_measurement_t m;
  dt_iop_gui_enter_critical_section(self);
  const gboolean have = g->auto_valid && g->auto_imgid == dev->image_storage.id;
  if(have) m = g->auto_m;
  g->auto_valid = FALSE;
  dt_iop_gui_leave_critical_section(self);
  if(!have) return;

  _store_measurement(p, &m);
  dt_dev_add_history_item(dev, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_filminversion_gui_data_t *g = IOP_GUI_ALLOC(filminversion);

  g->mode = dt_bauhaus_combobox_from_params(self, "mode");
  gtk_widget_set_tooltip_text(g->mode, _("select the film type: color negative (C-41)\n"
                                         "or black & white negative"));

  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "mask neutralization")));

  g->picker = dt_color_picker_new(self, DT_COLOR_PICKER_AREA, NULL);
  gtk_widget_set_tooltip_text(g->picker, _("measure the film's per-channel density range in the drawn\n"
                                           "area and keep it. the whole frame is measured once when\n"
                                           "the module is first applied; after that nothing is\n"
                                           "re-measured unless you use this picker. draw it over the\n"
                                           "picture only, leaving out the rebate and film holder."));
  dt_action_define_iop(self, N_("pickers"), N_("mask neutralize"), g->picker, &dt_action_def_toggle);
  dt_gui_box_add(self->widget, g->picker);

  g->mask_offset_R = dt_bauhaus_slider_from_params(self, "mask_offset[0]");
  dt_bauhaus_widget_set_label(g->mask_offset_R, NULL, N_("mask trim red"));
  g->mask_offset_G = dt_bauhaus_slider_from_params(self, "mask_offset[1]");
  dt_bauhaus_widget_set_label(g->mask_offset_G, NULL, N_("mask trim green"));
  g->mask_offset_B = dt_bauhaus_slider_from_params(self, "mask_offset[2]");
  dt_bauhaus_widget_set_label(g->mask_offset_B, NULL, N_("mask trim blue"));
  gtk_widget_set_tooltip_text(g->mask_offset_R, _("per-channel trim on top of the measured mask neutralization"));
  gtk_widget_set_tooltip_text(g->mask_offset_G, _("per-channel trim on top of the measured mask neutralization"));
  gtk_widget_set_tooltip_text(g->mask_offset_B, _("per-channel trim on top of the measured mask neutralization"));

  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "creative filtration")));

  g->filtration_R = dt_bauhaus_slider_from_params(self, "filtration[0]");
  dt_bauhaus_widget_set_label(g->filtration_R, NULL, N_("filtration red"));
  g->filtration_G = dt_bauhaus_slider_from_params(self, "filtration[1]");
  dt_bauhaus_widget_set_label(g->filtration_G, NULL, N_("filtration green"));
  g->filtration_B = dt_bauhaus_slider_from_params(self, "filtration[2]");
  dt_bauhaus_widget_set_label(g->filtration_B, NULL, N_("filtration blue"));
#define FILMINVERSION_FILTRATION_TOOLTIP _("additive density trim, like an enlarger dichroic-head color filter.\n" \
                                           "scaled by the measured per-channel range, so a slider unit\n" \
                                           "stays consistent across scans.")
  gtk_widget_set_tooltip_text(g->filtration_R, FILMINVERSION_FILTRATION_TOOLTIP);
  gtk_widget_set_tooltip_text(g->filtration_G, FILMINVERSION_FILTRATION_TOOLTIP);
  gtk_widget_set_tooltip_text(g->filtration_B, FILMINVERSION_FILTRATION_TOOLTIP);
#undef FILMINVERSION_FILTRATION_TOOLTIP

  dt_gui_box_add(self->widget, dt_ui_section_label_new(C_("section", "dye separation")));

  g->dye_separation_trim_R = dt_bauhaus_slider_from_params(self, "dye_separation_trim[0]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_R, NULL, N_("dye separation trim red"));
  g->dye_separation_trim_G = dt_bauhaus_slider_from_params(self, "dye_separation_trim[1]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_G, NULL, N_("dye separation trim green"));
  g->dye_separation_trim_B = dt_bauhaus_slider_from_params(self, "dye_separation_trim[2]");
  dt_bauhaus_widget_set_label(g->dye_separation_trim_B, NULL, N_("dye separation trim blue"));
#define FILMINVERSION_DYE_TOOLTIP _("density-domain saturation of this dye layer after the print curve.\n" \
                                    "positive separates it further from the other two, negative\n" \
                                    "pulls it toward them. neutrals are unchanged.")
  gtk_widget_set_tooltip_text(g->dye_separation_trim_R, FILMINVERSION_DYE_TOOLTIP);
  gtk_widget_set_tooltip_text(g->dye_separation_trim_G, FILMINVERSION_DYE_TOOLTIP);
  gtk_widget_set_tooltip_text(g->dye_separation_trim_B, FILMINVERSION_DYE_TOOLTIP);
#undef FILMINVERSION_DYE_TOOLTIP
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_R, 2);
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_G, 2);
  dt_bauhaus_slider_set_digits(g->dye_separation_trim_B, 2);

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished_callback);
}


void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_filminversion_gui_data_t *g = self->gui_data;
  const dt_iop_filminversion_params_t *const p = self->params;

  if(!w || w == g->mode)
  {
    // B&W collapses to one luminance density before any of these apply
    const gboolean color = p->mode != DT_FILMINVERSION_BW;
    gtk_widget_set_visible(g->mask_offset_R, color);
    gtk_widget_set_visible(g->mask_offset_G, color);
    gtk_widget_set_visible(g->mask_offset_B, color);
    gtk_widget_set_visible(g->filtration_R, color);
    gtk_widget_set_visible(g->filtration_G, color);
    gtk_widget_set_visible(g->filtration_B, color);
    gtk_widget_set_visible(g->dye_separation_trim_R, color);
    gtk_widget_set_visible(g->dye_separation_trim_G, color);
    gtk_widget_set_visible(g->dye_separation_trim_B, color);
  }
}


void gui_update(dt_iop_module_t *const self)
{
  dt_iop_color_picker_reset(self, TRUE);
  gui_changed(self, NULL, NULL);
}

void gui_reset(dt_iop_module_t *self)
{
  dt_iop_color_picker_reset(self, TRUE);
}
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
