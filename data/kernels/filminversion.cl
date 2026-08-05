/*
    This file is part of darktable,
    Copyright (C) 2024-2026 darktable developers.

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

#include "common.h"

/* Mirror of _process_pixel() in src/iop/filminversion.c -- the two must be
   kept in sync. Every per-frame quantity is solved on the host (see
   _frame_from_measurement) and arrives here as a scalar, so this kernel is
   pure per-pixel math with no reductions.

   The curve math and constants are adapted from NegPy
   (https://github.com/marcinz606/NegPy), GPLv3, reused here under GPLv3. */

#define FI_DENSITY_EPS 1e-10f
#define FI_PAPER_GAMMA_WIDTH 0.6f
#define FI_ZONE_SHARPNESS 4.0f
#define FI_ZONE_SHADOW_CENTER 1.5f
#define FI_ZONE_HIGHLIGHT_CENTER 0.35f

// Regional CMY: mirrors HD_CMY_ZONE_SHARPNESS / HD_ANCHOR_TARGET_DENSITY in
// filminversion.c. One shared zone center (unlike FI_ZONE_SHADOW/HIGHLIGHT_
// CENTER above), so the shadow/highlight weights always sum to 1.
#define FI_CMY_ZONE_SHARPNESS 3.0f
#define FI_ANCHOR_TARGET_DENSITY 0.75f

// Separation Damping: print-density chroma where the per-pixel gain crosses 1.0.
// Mirrors HD_SEPARATION_DAMPING_REF_SPREAD in filminversion.c.
#define FI_SEPARATION_DAMPING_REF_SPREAD 0.35f

#define FI_MODE_C41 0
#define FI_MODE_BW 1
#define FI_MODE_E6 2

// Numerically stable softplus: log(1 + exp(x)), algebraically identical to
// the CPU _softplus() branch but written with log1p. The naive
// log(1 + exp(x)) form loses nearly all precision for large negative x,
// where the argument sits just above 1.0 -- that cancellation was the
// dominant source of CPU/GPU divergence, so this stays on the precise
// builtins rather than the native_ approximations.
static inline float4 fi_softplus(const float4 x)
{
  return fmax(x, 0.0f) + log1p(exp(-fabs(x)));
}

static inline float4 fi_sigmoid(const float4 x)
{
  return 1.0f / (1.0f + exp(-x));
}

// 10^-D, mirroring the CPU's expf(-D * M_LN10)
static inline float4 fi_density_to_linear(const float4 density)
{
  return exp(-density * M_LN10_F);
}

// Black point compensation. Mirrors _apply_bpc() in filminversion.c.
// bpc_black is float4 (per-channel, C41/E6) -- B&W broadcasts its scalar
// bpc_black_bw into all four lanes at the call site instead of a separate
// scalar overload.
static inline float4 fi_apply_bpc(const float4 linear, const float4 bpc_black)
{
  return (linear - bpc_black) / (1.0f - bpc_black);
}

// a_sh/a_hl/d_min_eff/d_max_eff are float4 (per-channel curve shape, C41/E6)
// -- B&W broadcasts its scalar curve_bw components into all four lanes at
// the call site instead of a separate scalar overload.
static inline float4 fi_hd_curve(const float4 v, const float4 a_sh, const float4 a_hl,
                                 const float4 d_min_eff, const float4 d_max_eff)
{
  const float4 v1 = d_min_eff + fi_softplus(a_hl * (v - d_min_eff)) / a_hl;  // shoulder -> paper white
  return d_max_eff - fi_softplus(a_sh * (d_max_eff - v1)) / a_sh;            // toe -> paper black
}

// Snap, Regional CMY, Split Grade and Zone Density. Order matters: Snap
// reads the pre-everything v; Regional CMY (per-channel, unlike the other
// two) runs next, matching NegPy; Zone Density recomputes its weights on
// Split Grade's already-updated v. midtone_gamma/shadow_cmy/highlight_cmy
// are per-channel (float4) -- B&W broadcasts its single scalar into all
// four lanes at the call site instead of a separate scalar overload. Always
// computed rather than skipped when zero, since a per-lane "any nonzero"
// test isn't free in OpenCL C the way a scalar != is.
static inline float4 fi_snap_grade_density(float4 v, const float v_star,
                                           const float4 midtone_gamma,
                                           const float4 shadow_cmy, const float4 highlight_cmy,
                                           const float sh_grade, const float hl_grade,
                                           const float sh_density, const float hl_density)
{
  v = v + midtone_gamma * FI_PAPER_GAMMA_WIDTH * tanh((v - v_star) / FI_PAPER_GAMMA_WIDTH);

  {
    const float4 w_sh = fi_sigmoid(FI_CMY_ZONE_SHARPNESS * (v - FI_ANCHOR_TARGET_DENSITY));
    const float4 w_hi = 1.0f - w_sh;
    v = v + shadow_cmy * w_sh + highlight_cmy * w_hi;
  }

  if(sh_grade != 0.0f || hl_grade != 0.0f)
  {
    const float4 w_gsh = fi_sigmoid(FI_ZONE_SHARPNESS * (v - FI_ZONE_SHADOW_CENTER));
    const float4 w_ghi = 1.0f - fi_sigmoid(FI_ZONE_SHARPNESS * (v - FI_ZONE_HIGHLIGHT_CENTER));
    v = v + sh_grade * w_gsh * (v - FI_ZONE_SHADOW_CENTER)
          + hl_grade * w_ghi * (v - FI_ZONE_HIGHLIGHT_CENTER);
  }

  if(sh_density != 0.0f || hl_density != 0.0f)
  {
    const float4 w_zsh = fi_sigmoid(FI_ZONE_SHARPNESS * (v - FI_ZONE_SHADOW_CENTER));
    const float4 w_zhi = 1.0f - fi_sigmoid(FI_ZONE_SHARPNESS * (v - FI_ZONE_HIGHLIGHT_CENTER));
    v = v + sh_density * w_zsh + hl_density * w_zhi;
  }

  return v;
}

// Density-domain saturation + Separation Damping. Mirrors
// _apply_dye_separation() in filminversion.c; operates on print density.
// base/k are per-channel (NegPy per_channel_dye_separation trims); chroma is
// shared (one property of the pixel, not per-channel). Always computed --
// k=(1,1,1) is a near no-op, and a per-lane "any k>1" test isn't free in
// OpenCL C the way a scalar != is, so the CPU side gates on that at the call
// site instead of inside this function.
static inline float4 fi_apply_dye_separation(float4 D, const float4 base,
                                              const float4 k, const float damping)
{
  float3 e = D.xyz - base.xyz;
  const float e_mean = (e.x + e.y + e.z) / 3.0f;

  if(damping <= 0.0f)
    return (float4)(base.xyz + e_mean + k.xyz * (e - e_mean), D.w);

  const float3 dd = e - e_mean;
  const float chroma = sqrt(((dd.x - dd.y) * (dd.x - dd.y) + (dd.y - dd.z) * (dd.y - dd.z) + (dd.x - dd.z) * (dd.x - dd.z)) / 3.0f);
  const float h = (FI_SEPARATION_DAMPING_REF_SPREAD - chroma) / max(FI_SEPARATION_DAMPING_REF_SPREAD + chroma, 1e-6f);
  const float exponent = (1.0f - damping) + damping * h;
  const float3 gain = min(pow(k.xyz, (float3)exponent), (float3)3.0f);

  return (float4)(base.xyz + e_mean + gain * dd, D.w);
}

/* frame_bw packs the luma-combined B&W equivalents as
     (floor_bw, ceil_bw, slope_bw, pivot_bw)
   curve_bw packs the shared/scalar curve shape B&W uses (no per-channel
     trims there): (a_sh_bw, a_hl_bw, d_min_eff_bw, d_max_eff_bw)
   a_sh/a_hl/d_min_eff/d_max_eff/bpc_black (own float4 args below) are the
     per-channel, grade-coupled-and-trimmed shape C41/E6 uses instead
   tone  packs (v_star, midtone_gamma [B&W only], shadow_grade_gain, highlight_grade_gain)
   midtone_gamma_c is Snap's per-channel (trimmed) value, C41/E6 only
   zone  packs (shadow_density, highlight_density, anchor_scale, exposure) */
kernel void
filminversion(read_only image2d_t in, write_only image2d_t out,
              const int width, const int height, const int mode,
              const float4 frame_floor, const float4 frame_ceil,
              const float4 frame_slope, const float4 frame_pivot,
              const float4 frame_curv, const float4 frame_filtration,
              const float4 mask_offset, const float4 frame_bw,
              const float4 curve_bw,
              const float4 curve_a_sh, const float4 curve_a_hl,
              const float4 curve_d_min_eff, const float4 curve_d_max_eff,
              const float4 bpc_black, const float bpc_black_bw,
              const float4 tone, const float4 midtone_gamma_c, const float4 zone,
              const float4 shadow_filtration, const float4 highlight_filtration,
              const float4 dye_k, const float separation_damping)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 pix = readpixel(in, x, y);

  const float v_star = tone.x;
  const float sh_grade = tone.z, hl_grade = tone.w;
  const float sh_density = zone.x, hl_density = zone.y;
  const float out_scale = zone.z * zone.w; // anchor_scale * exposure

  float4 res;

  if(mode == FI_MODE_BW)
  {
    // B&W: one luminance density, no mask neutralization, filtration, or
    // per-channel trims -- uses the shared/scalar curve shape (curve_bw,
    // bpc_black_bw) and tone.y (untrimmed midtone_gamma) throughout.
    const float luma = fmax(0.2126f * pix.x + 0.7152f * pix.y + 0.0722f * pix.z, FI_DENSITY_EPS);
    const float density = -log10(luma);
    const float range = fmax(frame_bw.y - frame_bw.x, 1e-3f);
    const float normalized = (density - frame_bw.x) / range;
    // invert sign for the log-exposure axis; v_star is already baked into
    // pivot_bw, so no separate "+ v_star" here
    const float x_adj = -(normalized - frame_bw.w);

    float4 v = (float4)(frame_bw.z * x_adj);
    // B&W has no per-channel color, so Regional CMY is zeroed -- otherwise
    // the .x lane used for output below would pick up the "red" tint alone.
    v = fi_snap_grade_density(v, v_star, (float4)tone.y, (float4)0.0f, (float4)0.0f, sh_grade, hl_grade, sh_density, hl_density);
    const float4 d_print = fi_hd_curve(v, (float4)curve_bw.x, (float4)curve_bw.y, (float4)curve_bw.z, (float4)curve_bw.w);
    const float o = out_scale * fi_apply_bpc(fi_density_to_linear(d_print), (float4)bpc_black_bw).x;
    res = (float4)(o, o, o, pix.w);
  }
  else
  {
    // C41 (negative, invert sign) or E6 (slide, do not invert sign)
    const float invert_sign = (mode == FI_MODE_E6) ? 1.0f : -1.0f;

    const float4 density = -(float4)(log10(fmax(pix.x, FI_DENSITY_EPS)),
                                     log10(fmax(pix.y, FI_DENSITY_EPS)),
                                     log10(fmax(pix.z, FI_DENSITY_EPS)),
                                     0.0f);

    // the automatic per-channel normalization IS the mask neutralization;
    // mask_offset/filtration are optional manual trims on top
    const float4 range = fmax(frame_ceil - frame_floor, (float4)1e-3f);
    const float4 normalized = (density - frame_floor) / range - mask_offset + frame_filtration;

    const float4 x_adj = invert_sign * (normalized - frame_pivot);

    // frame_slope already includes k_trim and frame_pivot already bakes in
    // v_star; frame_curv is 0 unless Cast Removal is active.
    float4 v = frame_slope * x_adj + frame_curv * normalized * normalized;
    v = fi_snap_grade_density(v, v_star, midtone_gamma_c, shadow_filtration, highlight_filtration, sh_grade, hl_grade, sh_density, hl_density);
    float4 d_print = fi_hd_curve(v, curve_a_sh, curve_a_hl, curve_d_min_eff, curve_d_max_eff);

    // Dye Separation + Separation Damping: density-domain saturation on the
    // print curve output. dye_k=(1,1,1) is a near no-op; always computed
    // (the CPU path gates this call on "is any channel's k > 1", but a
    // per-lane test isn't free here the way a scalar != is).
    d_print = fi_apply_dye_separation(d_print, curve_d_min_eff, dye_k, separation_damping);

    res = out_scale * fi_apply_bpc(fi_density_to_linear(d_print), bpc_black);
    res.w = pix.w;
  }

  write_imagef(out, (int2)(x, y), res);
}
