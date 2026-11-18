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

#define FI_MODE_C41 0
#define FI_MODE_BW 1

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

// 10^-D, mirroring the CPU's expf(-D * M_LN10)
static inline float4 fi_density_to_linear(const float4 density)
{
  return exp(-density * M_LN10_F);
}

// curve packs (a_sh, a_hl, d_min_eff, d_max_eff), shared by every channel
static inline float4 fi_hd_curve(const float4 v, const float4 curve)
{
  const float4 v1 = curve.z + fi_softplus(curve.y * (v - curve.z)) / curve.y;  // shoulder -> paper white
  return curve.w - fi_softplus(curve.x * (curve.w - v1)) / curve.x;            // toe -> paper black
}

/* frame_bw packs the luma-combined B&W equivalents as
     (floor_bw, ceil_bw, slope_bw, pivot_bw)
   out_scale is the mid-gray anchor_scale */
kernel void
filminversion(read_only image2d_t in, write_only image2d_t out,
              const int width, const int height, const int mode,
              const float4 frame_floor, const float4 frame_ceil,
              const float4 frame_slope, const float4 frame_pivot,
              const float4 frame_curv, const float4 frame_filtration,
              const float4 mask_offset, const float4 frame_bw, const float4 curve,
              const float bpc_black, const float out_scale,
              const float4 dye_k, const int dye_separation)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 pix = readpixel(in, x, y);

  float4 res;

  if(mode == FI_MODE_BW)
  {
    // B&W: one luminance density, no mask neutralization or filtration
    const float luma = fmax(0.2126f * pix.x + 0.7152f * pix.y + 0.0722f * pix.z, FI_DENSITY_EPS);
    const float density = -log10(luma);
    const float range = fmax(frame_bw.y - frame_bw.x, 1e-3f);
    const float normalized = (density - frame_bw.x) / range;
    // invert sign for the log-exposure axis; v_star is already baked into
    // pivot_bw, so no separate "+ v_star" here
    const float4 v = (float4)(frame_bw.z * -(normalized - frame_bw.w));
    const float4 d_print = fi_hd_curve(v, curve);
    const float o = out_scale * (fi_density_to_linear(d_print).x - bpc_black) / (1.0f - bpc_black);
    res = (float4)(o, o, o, pix.w);
  }
  else
  {
    const float4 density = -(float4)(log10(fmax(pix.x, FI_DENSITY_EPS)),
                                     log10(fmax(pix.y, FI_DENSITY_EPS)),
                                     log10(fmax(pix.z, FI_DENSITY_EPS)),
                                     0.0f);

    // the measured per-channel normalization IS the mask neutralization;
    // mask_offset/filtration are manual trims on top
    const float4 range = fmax(frame_ceil - frame_floor, (float4)1e-3f);
    const float4 normalized = (density - frame_floor) / range - mask_offset + frame_filtration;

    // negative: invert the log-exposure axis. frame_pivot already bakes in
    // v_star; frame_curv is 0 unless Cast Removal found a highlight reference.
    const float4 v = frame_slope * -(normalized - frame_pivot) + frame_curv * normalized * normalized;
    float4 d_print = fi_hd_curve(v, curve);

    // dye separation: k*I + (1-k)*J around the pixel's mean density, see
    // _apply_dye_separation()
    if(dye_separation)
    {
      const float mean = (d_print.x + d_print.y + d_print.z) / 3.0f;
      d_print.xyz = mean + dye_k.xyz * (d_print.xyz - mean);
    }

    res = out_scale * (fi_density_to_linear(d_print) - bpc_black) / (1.0f - bpc_black);
    res.w = pix.w;
  }

  write_imagef(out, (int2)(x, y), res);
}
