/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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
#include "common/colorspaces_inline_conversions.h"
#include "common/interpolation.h"
#include "common/iop_order.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/opencl.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/tiling.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <stdlib.h>

/*
  radial distortion model, centered on the image and normalized to the
  half-diagonal. an output (corrected) pixel at radius x samples the
  input (distorted) image at radius

    g(x) = x * (1 + k1 x^2 + k2 x^4 + k3 x^6)

  the slider drives k1 only, which is enough for barrel and pincushion
  distortion. k2 and k3 are kept in the params for moustache distortion.

  lateral chromatic aberration is a magnification that differs per
  channel. red and blue sample the input at s(r) times the radius green
  samples it at, with r the normalized input radius and

    s(r) = 1 + a + b r^2

  a and b are measured from the image, against green, the first time
  the pipe runs with the correction on, and are then stored in the params
  like the lens module's TCA values.
*/

DT_MODULE_INTROSPECTION(2, dt_iop_distortion_params_t)

// positive slider values correct barrel distortion, which needs k1 < 0.
// at +-100, |k1| = 0.25 keeps g() monotonic up to x = 1.15, i.e. past
// the image corners, with room for the auto-scale to zoom out
#define DT_DISTORTION_SLIDER_TO_K1 (-0.0025f)

typedef struct dt_iop_distortion_params_t
{
  float distortion; // $MIN: -100.0 $MAX: 100.0 $DEFAULT: 0.0 $DESCRIPTION: "distortion"
  float k2;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "4th order"
  float k3;         // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "6th order"
  gboolean autoscale; // $DEFAULT: TRUE $DESCRIPTION: "auto-scale"
  gboolean ca_correct; // $DEFAULT: TRUE $DESCRIPTION: "correct chromatic aberration"
  gboolean ca_measured; // $DEFAULT: FALSE
  float ca_red_a;   // $DEFAULT: 0.0
  float ca_red_b;   // $DEFAULT: 0.0
  float ca_blue_a;  // $DEFAULT: 0.0
  float ca_blue_b;  // $DEFAULT: 0.0
} dt_iop_distortion_params_t;

// largest accepted |s(r) - 1| at the center and the corners. lateral CA
// rarely reaches 0.1%, so anything past this is a failed measurement
#define DT_DISTORTION_CA_MAX 0.005f

// the measurement runs on at most this many pixels along the long edge,
// so that a full size export measures at about the preview pipe's scale
#define DT_DISTORTION_CA_SIZE 2000

typedef struct dt_iop_distortion_data_t
{
  float k1, k2, k3;
  gboolean autoscale;
  gboolean ca_correct;  // red and blue get their own scale
  gboolean ca_measure;  // the coefficients are not known yet
  float ca[4];          // red a, red b, blue a, blue b
} dt_iop_distortion_data_t;

// the gui data is needed for dt_iop_gui_update() to refresh the
// _from_params widgets (src/develop/imageop.c), and it carries the
// measurement from the preview pipe to the history
typedef struct dt_iop_distortion_gui_data_t
{
  GtkWidget *distortion, *autoscale, *ca_correct;
  gboolean ca_ready;    // ca holds a measurement not yet stored
  dt_imgid_t ca_imgid;  // of the image it was made on
  float ca[4];
} dt_iop_distortion_gui_data_t;

typedef struct dt_iop_distortion_global_data_t
{
  int kernel_manual_distortion;
} dt_iop_distortion_global_data_t;


const char *name()
{
  return C_("modulename", "manual distortion");
}

const char *aliases()
{
  return _("barrel|pincushion|lens distortion");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("manually correct barrel and pincushion lens distortion"),
                                      _("corrective"),
                                      _("linear, RGB, scene-referred"),
                                      _("geometric, RGB"),
                                      _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_CORRECT | IOP_GROUP_TECHNICAL;
}

int operation_tags()
{
  return IOP_TAG_DISTORT;
}

int flags()
{
  return IOP_FLAGS_ALLOW_TILING | IOP_FLAGS_TILING_FULL_ROI | IOP_FLAGS_GUIDES_WIDGET;
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
  if(old_version == 1)
  {
    typedef struct dt_iop_distortion_params_v1_t
    {
      float distortion;
      float k2;
      float k3;
      gboolean autoscale;
    } dt_iop_distortion_params_v1_t;

    const dt_iop_distortion_params_v1_t *o = old_params;
    dt_iop_distortion_params_t *n = calloc(1, sizeof(dt_iop_distortion_params_t));
    n->distortion = o->distortion;
    n->k2 = o->k2;
    n->k3 = o->k3;
    n->autoscale = o->autoscale;
    // keep existing edits looking as they did
    n->ca_correct = FALSE;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_distortion_params_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

// one channel of a 4-channel buffer, bilinearly interpolated and clamped
// to the buffer
static inline float _bilinear(const float *const in,
                              const int width,
                              const int height,
                              const float x,
                              const float y,
                              const int c)
{
  const float xc = CLAMP(x, 0.0f, width - 1.001f);
  const float yc = CLAMP(y, 0.0f, height - 1.001f);
  const int ix = xc;
  const int iy = yc;
  const float fx = xc - ix;
  const float fy = yc - iy;
  const float *const p = in + 4 * ((size_t)iy * width + ix);
  const float *const q = p + 4 * width;
  return (1.0f - fy) * ((1.0f - fx) * p[c] + fx * p[4 + c])
         + fy * ((1.0f - fx) * q[c] + fx * q[4 + c]);
}

// the radial shift d, in pixels, that best maps channel c onto green
// over one tile: c(p + d u(p)) ~ green(p), with u(p) the unit vector
// pointing away from the center (cx, cy). both channels are
// standardized over the tile, so a gain or offset between them does not
// matter, and Gauss-Newton refines d from 0. returns FALSE when the
// tile has no edge that tells
static gboolean _tile_shift(const float *const buf,
                            const int width,
                            const int height,
                            const int x0,
                            const int y0,
                            const int size,
                            const float cx,
                            const float cy,
                            const int c,
                            float *const shift,
                            float *const weight)
{
  const double n = (double)size * size;

  double sg = 0.0, sgg = 0.0;
  for(int j = y0; j < y0 + size; j++)
    for(int i = x0; i < x0 + size; i++)
    {
      const double g = buf[4 * ((size_t)j * width + i) + 1];
      sg += g;
      sgg += g * g;
    }
  const double mg = sg / n;
  const double sdg = sqrt(MAX(sgg / n - mg * mg, 0.0));
  // flat or noise-only tiles
  if(sdg < 0.02 * fabs(mg) + 1e-6) return FALSE;

  const float h = 0.5f; // central difference step
  float d = 0.0f;
  double ncc = 0.0, sjj = 0.0, stt = 0.0;
  for(int it = 0; it < 8; it++)
  {
    double sc = 0.0, scc = 0.0, sj = 0.0, sjc = 0.0, sjg = 0.0, scg = 0.0;
    sjj = stt = 0.0;
    for(int j = y0; j < y0 + size; j++)
      for(int i = x0; i < x0 + size; i++)
      {
        const float dx = i - cx;
        const float dy = j - cy;
        const float r = MAX(dt_fast_hypotf(dx, dy), 1e-3f);
        const float ux = dx / r;
        const float uy = dy / r;
        const float x = i + d * ux;
        const float y = j + d * uy;
        const double v = _bilinear(buf, width, height, x, y, c);
        const double jr = (_bilinear(buf, width, height, x + h * ux, y + h * uy, c)
                           - _bilinear(buf, width, height, x - h * ux, y - h * uy, c)) / (2.0f * h);
        const double jt = (_bilinear(buf, width, height, x - h * uy, y + h * ux, c)
                           - _bilinear(buf, width, height, x + h * uy, y - h * ux, c)) / (2.0f * h);
        const double g = (buf[4 * ((size_t)j * width + i) + 1] - mg) / sdg;
        sc += v;
        scc += v * v;
        sj += jr;
        sjj += jr * jr;
        stt += jt * jt;
        sjc += jr * v;
        sjg += jr * g;
        scg += v * g;
      }
    const double mc = sc / n;
    const double sdc = sqrt(MAX(scc / n - mc * mc, 0.0));
    if(sdc < 1e-9 || sjj < 1e-12) return FALSE;

    // the green term is standardized, the channel one not yet
    ncc = scg / (sdc * n);
    double step = (sdc * sjg - (sjc - mc * sj)) / sjj;
    step = CLAMP(step, -0.5, 0.5);
    d += step;
    if(fabsf(d) > 2.0f) return FALSE;
    if(fabs(step) < 1e-3) break;
  }

  // a measurement needs edges across the radius and two channels that
  // agree on them; colored edges and noise fail the correlation
  if(ncc < 0.9 || sjj < 0.25 * (sjj + stt)) return FALSE;

  *shift = d;
  *weight = sjj;
  return TRUE;
}

// weighted least squares fit of d(r) = a r + b r^3, refitted after
// dropping the outliers
static gboolean _fit_shift(const float *const r,
                           const float *const d,
                           float *const w,
                           const int n,
                           float *const a,
                           float *const b)
{
  for(int pass = 0; pass < 4; pass++)
  {
    double s11 = 0.0, s12 = 0.0, s22 = 0.0, t1 = 0.0, t2 = 0.0, sw = 0.0;
    int count = 0;
    for(int i = 0; i < n; i++)
    {
      if(w[i] <= 0.0f) continue;
      const double r2 = (double)r[i] * r[i];
      s11 += w[i] * r2;
      s12 += w[i] * r2 * r2;
      s22 += w[i] * r2 * r2 * r2;
      t1 += w[i] * r[i] * d[i];
      t2 += w[i] * r2 * r[i] * d[i];
      sw += w[i];
      count++;
    }
    if(count < 16) return FALSE;

    // tiles over a narrow range of radii cannot tell a from b
    const double det = s11 * s22 - s12 * s12;
    if(det > 1e-3 * s11 * s22)
    {
      *a = (t1 * s22 - t2 * s12) / det;
      *b = (s11 * t2 - s12 * t1) / det;
    }
    else
    {
      *a = t1 / s11;
      *b = 0.0f;
    }
    if(pass == 3) break;

    double sr = 0.0;
    for(int i = 0; i < n; i++)
    {
      if(w[i] <= 0.0f) continue;
      const float res = d[i] - r[i] * (*a + *b * r[i] * r[i]);
      sr += w[i] * res * res;
    }
    const float sigma = sqrt(sr / sw);
    for(int i = 0; i < n; i++)
    {
      if(w[i] <= 0.0f) continue;
      const float res = d[i] - r[i] * (*a + *b * r[i] * r[i]);
      if(fabsf(res) > 2.5f * sigma) w[i] = 0.0f;
    }
  }
  return TRUE;
}

// measures the lateral CA of in, which covers roi_in of an image
// centered at (w2, h2), as the a and b of s(r) for red then blue, in the
// channels of the transposed matrix to_in_t times the pixels. leaves
// zeros, i.e. no correction, when the image does not have enough edges
static void _measure_ca(const float *const in,
                        const dt_iop_roi_t *const roi_in,
                        const float w2,
                        const float h2,
                        const dt_colormatrix_t to_in_t,
                        float ca[4])
{
  for(int k = 0; k < 4; k++) ca[k] = 0.0f;

  // box downsampling moves no channel against another
  const int f = MAX(1, (int)ceilf(MAX(roi_in->width, roi_in->height)
                                  / (float)DT_DISTORTION_CA_SIZE));
  const int width = roi_in->width / f;
  const int height = roi_in->height / f;
  if(width < 128 || height < 128) return;

  float *const src = dt_alloc_align_float((size_t)4 * width * height);
  if(!src) return;
  const float norm = 1.0f / (f * f);
  DT_OMP_FOR()
  for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
      dt_aligned_pixel_t sum = { 0.0f };
      for(int v = 0; v < f; v++)
        for(int u = 0; u < f; u++)
        {
          const float *const p = in + 4 * ((size_t)(f * j + v) * roi_in->width + f * i + u);
          for_each_channel(c) sum[c] += norm * p[c];
        }
      dt_apply_transposed_color_matrix(sum, to_in_t, src + 4 * ((size_t)j * width + i));
    }

  const float cx = (w2 - roi_in->x - 0.5f * (f - 1)) / f;
  const float cy = (h2 - roi_in->y - 0.5f * (f - 1)) / f;
  const float rnorm = dt_fast_hypotf(w2, h2) / f;

  const int size = CLAMP(MIN(width, height) / 24, 16, 48);
  const int margin = 4;
  const int ntx = (width - 2 * margin) / size;
  const int nty = (height - 2 * margin) / size;
  const int nt = ntx * nty;

  float *const r = calloc(nt, sizeof(float));
  float *const d = calloc(2 * (size_t)nt, sizeof(float));
  float *const w = calloc(2 * (size_t)nt, sizeof(float));
  if(r && d && w)
  {
    DT_OMP_FOR()
    for(int t = 0; t < nt; t++)
    {
      const int x0 = margin + (t % ntx) * size;
      const int y0 = margin + (t / ntx) * size;
      r[t] = dt_fast_hypotf(x0 + 0.5f * size - cx, y0 + 0.5f * size - cy) / rnorm;
      // near the center the shift is too small to measure, and the
      // radial direction changes across the tile
      if(r[t] < 0.2f) continue;
      for(int k = 0; k < 2; k++)
      {
        float shift, weight;
        if(_tile_shift(src, width, height, x0, y0, size, cx, cy, 2 * k, &shift, &weight))
        {
          d[k * nt + t] = shift / rnorm;
          w[k * nt + t] = weight;
        }
      }
    }

    for(int k = 0; k < 2; k++)
    {
      float a, b;
      if(_fit_shift(r, d + k * nt, w + k * nt, nt, &a, &b)
         && fabsf(a) < DT_DISTORTION_CA_MAX && fabsf(a + b) < DT_DISTORTION_CA_MAX)
      {
        ca[2 * k] = a;
        ca[2 * k + 1] = b;
      }
    }
  }

  dt_print(DT_DEBUG_PIPE, "[distortion] lateral CA on %dx%d: red %.6f %.6f, blue %.6f %.6f",
           width, height, ca[0], ca[1], ca[2], ca[3]);

  free(r);
  free(d);
  free(w);
  dt_free_align(src);
}

// keeps the measurement for the history when it was made by the preview
// pipe, which sees the whole image at any zoom
static void _ca_measured(dt_iop_module_t *self,
                         dt_dev_pixelpipe_iop_t *piece,
                         const float ca[4])
{
  dt_iop_distortion_gui_data_t *g = self->gui_data;
  if(!g || !dt_pipe_is_preview(piece->pipe)) return;

  dt_iop_gui_enter_critical_section(self);
  for(int k = 0; k < 4; k++) g->ca[k] = ca[k];
  g->ca_imgid = piece->pipe->image.id;
  g->ca_ready = TRUE;
  dt_iop_gui_leave_critical_section(self);
}

// lateral CA shifts the sensor's or the file's own channels. raw files
// reach this module before colorin, in camera RGB, but other files come
// in the work profile, whose red and blue mix in some green. to_in_t and
// to_work_t, transposed, convert to and from the input profile's linear
// RGB, or are the identity when that is not needed or not possible
static void _ca_space(const dt_iop_module_t *self,
                      const dt_dev_pixelpipe_iop_t *piece,
                      dt_colormatrix_t to_in_t,
                      dt_colormatrix_t to_work_t)
{
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 4; j++)
      to_in_t[i][j] = to_work_t[i][j] = (i == j && i < 3) ? 1.0f : 0.0f;

  if(self->iop_order < dt_ioppr_get_iop_order(self->dev->iop_order_list, "colorin", 0))
    return;

  const dt_iop_order_iccprofile_info_t *const work
      = dt_ioppr_get_pipe_work_profile_info(piece->pipe);
  const dt_iop_order_iccprofile_info_t *const input
      = dt_ioppr_get_pipe_input_profile_info(piece->pipe);
  if(!work || !input
     || !dt_is_valid_colormatrix(work->matrix_in[0][0])
     || !dt_is_valid_colormatrix(work->matrix_out[0][0])
     || !dt_is_valid_colormatrix(input->matrix_in[0][0])
     || !dt_is_valid_colormatrix(input->matrix_out[0][0]))
    return;

  dt_colormatrix_t to_in = { { 0.0f } };
  dt_colormatrix_t to_work = { { 0.0f } };
  dt_colormatrix_mul(to_in, input->matrix_out, work->matrix_in);
  dt_colormatrix_mul(to_work, work->matrix_out, input->matrix_in);
  dt_colormatrix_transpose(to_in_t, to_in);
  dt_colormatrix_transpose(to_work_t, to_work);
}

// s(r) for red and blue at normalized input radius r = sqrt(r2)
static inline void _ca_scales(const float ca[4],
                              const float r2,
                              float *const sr,
                              float *const sb)
{
  *sr = 1.0f + ca[0] + ca[1] * r2;
  *sb = 1.0f + ca[2] + ca[3] * r2;
}

// bounds of s(r) over the image, for the roi; s is monotonic in r^2
static void _ca_bounds(const dt_iop_distortion_data_t *const d,
                       float *const lo,
                       float *const hi)
{
  *lo = *hi = 1.0f;
  if(!d->ca_correct) return;
  if(d->ca_measure)
  {
    *lo = 1.0f - DT_DISTORTION_CA_MAX;
    *hi = 1.0f + DT_DISTORTION_CA_MAX;
    return;
  }
  float s[4];
  _ca_scales(d->ca, 0.0f, &s[0], &s[1]);
  _ca_scales(d->ca, 1.0f, &s[2], &s[3]);
  for(int k = 0; k < 4; k++)
  {
    *lo = MIN(*lo, s[k]);
    *hi = MAX(*hi, s[k]);
  }
}

// g(x) / x, from x^2
static inline float _ratio(const dt_iop_distortion_data_t *const d, const float x2)
{
  return 1.0f + x2 * (d->k1 + x2 * (d->k2 + x2 * d->k3));
}

// g'(x), from x^2
static inline float _slope(const dt_iop_distortion_data_t *const d, const float x2)
{
  return 1.0f + x2 * (3.0f * d->k1 + x2 * (5.0f * d->k2 + x2 * 7.0f * d->k3));
}

// with the output scaled by 1/t, does every border point of the output
// sample inside the input? output and input share the same rectangle, so
// along each ray the border sits at the same normalized radius rho in
// both, and rho covers [srr, 1] over the whole border
static gboolean _fits(const dt_iop_distortion_data_t *const d,
                      const float srr,
                      const float t)
{
  const int samples = 64;
  for(int i = 0; i < samples; i++)
  {
    const float f = (float)i / (float)(samples - 1);
    const float rho = srr + (1.0f - srr) * f;
    const float x = t * rho;
    if(x * _ratio(d, x * x) > rho) return FALSE;

    // g() must also be monotonic up to the corners, or the inside of
    // a ray folds over and maps outside the input
    const float y = t * f;
    if(_slope(d, y * y) <= 0.0f) return FALSE;
  }
  return TRUE;
}

// largest t for which the corrected image has no empty area
static float _get_scale(const dt_iop_distortion_data_t *const d,
                        const float width,
                        const float height)
{
  if(!d->autoscale || width < 1.0f || height < 1.0f) return 1.0f;

  const float srr = MIN(width, height) / dt_fast_hypotf(width, height);

  float lo = 0.25f;
  float hi = 2.0f;
  if(_fits(d, srr, hi)) return hi;
  if(!_fits(d, srr, lo)) return lo;

  // the fitting t form an interval starting at 0, as g() is monotonic
  for(int k = 0; k < 30; k++)
  {
    const float mid = 0.5f * (lo + hi);
    if(_fits(d, srr, mid))
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

gboolean distort_transform(dt_iop_module_t *self,
                           dt_dev_pixelpipe_iop_t *piece,
                           float *const restrict points,
                           size_t points_count)
{
  const dt_iop_distortion_data_t *d = piece->data;

  const float w2 = 0.5f * piece->buf_in.width;
  const float h2 = 0.5f * piece->buf_in.height;
  const float inv_r = 1.0f / dt_fast_hypotf(w2, h2);
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);

  for(size_t i = 0; i < 2 * points_count; i += 2)
  {
    const float cx = points[i] - w2;
    const float cy = points[i + 1] - h2;
    const float rd = inv_r * dt_fast_hypotf(cx, cy);
    if(rd <= 0.0f) continue;

    // invert g() with Newton's method, starting from the identity
    float x = rd;
    for(int k = 0; k < 10; k++)
    {
      const float x2 = x * x;
      const float slope = _slope(d, x2);
      if(slope <= 0.0f) break;
      const float step = (x * _ratio(d, x2) - rd) / slope;
      x -= step;
      if(fabsf(step) < 1e-6f) break;
    }

    const float m = x / (t * rd);
    points[i] = w2 + m * cx;
    points[i + 1] = h2 + m * cy;
  }

  return TRUE;
}

gboolean distort_backtransform(dt_iop_module_t *self,
                               dt_dev_pixelpipe_iop_t *piece,
                               float *const restrict points,
                               size_t points_count)
{
  const dt_iop_distortion_data_t *d = piece->data;

  const float w2 = 0.5f * piece->buf_in.width;
  const float h2 = 0.5f * piece->buf_in.height;
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);
  const float tr = t / dt_fast_hypotf(w2, h2);
  const float tr2 = tr * tr;

  for(size_t i = 0; i < 2 * points_count; i += 2)
  {
    const float cx = points[i] - w2;
    const float cy = points[i + 1] - h2;
    const float m = t * _ratio(d, tr2 * (cx * cx + cy * cy));
    points[i] = w2 + m * cx;
    points[i + 1] = h2 + m * cy;
  }

  return TRUE;
}

void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *const roi_out,
                   dt_iop_roi_t *roi_in)
{
  const dt_iop_distortion_data_t *d = piece->data;

  *roi_in = *roi_out;

  const float orig_w = roi_in->scale * piece->buf_in.width;
  const float orig_h = roi_in->scale * piece->buf_in.height;
  const float w2 = 0.5f * orig_w;
  const float h2 = 0.5f * orig_h;
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);
  const float tr = t / dt_fast_hypotf(w2, h2);
  const float tr2 = tr * tr;

  float xm = FLT_MAX, xM = -FLT_MAX, ym = FLT_MAX, yM = -FLT_MAX;

  // red and blue sample at up to s_lo..s_hi times the green position
  float s_lo, s_hi;
  _ca_bounds(d, &s_lo, &s_hi);

  // g() is monotonic, so the sources of the roi border bound the
  // sources of its inside
  const int width = roi_out->width;
  const int height = roi_out->height;
  const int perimeter = 2 * (width + height);
  for(int k = 0; k < perimeter; k++)
  {
    float x, y;
    if(k < width)
    {
      x = k;
      y = 0;
    }
    else if(k < 2 * width)
    {
      x = k - width;
      y = height - 1;
    }
    else if(k < 2 * width + height)
    {
      x = 0;
      y = k - 2 * width;
    }
    else
    {
      x = width - 1;
      y = k - 2 * width - height;
    }

    const float cx = roi_out->x + x - w2;
    const float cy = roi_out->y + y - h2;
    const float m = t * _ratio(d, tr2 * (cx * cx + cy * cy));
    xm = MIN(xm, w2 + m * MIN(s_lo * cx, s_hi * cx));
    xM = MAX(xM, w2 + m * MAX(s_lo * cx, s_hi * cx));
    ym = MIN(ym, h2 + m * MIN(s_lo * cy, s_hi * cy));
    yM = MAX(yM, h2 + m * MAX(s_lo * cy, s_hi * cy));
  }

  const dt_interpolation_t *interpolation = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);
  const float iw1 = interpolation->width;
  const float iw2 = 2.0f * iw1;
  roi_in->x = xm - iw1;
  roi_in->y = ym - iw1;
  roi_in->width = xM + iw2 - xm + 1.0f;
  roi_in->height = yM + iw2 - ym + 1.0f;

  roi_in->x = CLAMP(roi_in->x, 0, (int)floorf(orig_w - 2.0f));
  roi_in->y = CLAMP(roi_in->y, 0, (int)floorf(orig_h - 2.0f));
  roi_in->width = CLAMP(roi_in->width, 1, (int)floorf(orig_w) - roi_in->x);
  roi_in->height = CLAMP(roi_in->height, 1, (int)floorf(orig_h) - roi_in->y);
}

void distort_mask(dt_iop_module_t *self,
                  dt_dev_pixelpipe_iop_t *piece,
                  const float *const in,
                  float *const out,
                  const dt_iop_roi_t *const roi_in,
                  const dt_iop_roi_t *const roi_out)
{
  const dt_iop_distortion_data_t *d = piece->data;

  const float orig_w = roi_in->scale * piece->buf_in.width;
  const float orig_h = roi_in->scale * piece->buf_in.height;
  const float w2 = 0.5f * orig_w;
  const float h2 = 0.5f * orig_h;
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);
  const float tr = t / dt_fast_hypotf(w2, h2);
  const float tr2 = tr * tr;

  const float limw = roi_in->width - 1;
  const float limh = roi_in->height - 1;

  const dt_interpolation_t *interpolation = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < roi_out->height; y++)
  {
    for(int x = 0; x < roi_out->width; x++)
    {
      const float cx = roi_out->x + x - w2;
      const float cy = roi_out->y + y - h2;
      const float m = t * _ratio(d, tr2 * (cx * cx + cy * cy));
      const float xs = w2 + m * cx;
      const float ys = h2 + m * cy;
      const size_t idx = (size_t)y * roi_out->width + x;

      if(xs < -0.5f || ys < -0.5f || xs > orig_w - 0.5f || ys > orig_h - 0.5f)
      {
        out[idx] = 0.0f;
        continue;
      }

      const float xi = CLAMP(xs - roi_in->x, 0.0f, limw);
      const float yi = CLAMP(ys - roi_in->y, 0.0f, limh);
      out[idx] = CLIP(dt_interpolation_compute_sample(interpolation, in, xi, yi,
                                                      roi_in->width, roi_in->height,
                                                      1, roi_in->width));
    }
  }
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_distortion_data_t *d = piece->data;
  const float *const restrict in = (const float *)ivoid;
  float *const restrict out = (float *)ovoid;

  const float orig_w = roi_in->scale * piece->buf_in.width;
  const float orig_h = roi_in->scale * piece->buf_in.height;
  const float w2 = 0.5f * orig_w;
  const float h2 = 0.5f * orig_h;
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);
  const float tr = t / dt_fast_hypotf(w2, h2);
  const float tr2 = tr * tr;

  const float limw = roi_in->width - 1;
  const float limh = roi_in->height - 1;

  const gboolean ca_on = d->ca_correct;
  float ca[4] = { 0.0f };
  dt_colormatrix_t to_in_t, to_work_t;
  _ca_space(self, piece, to_in_t, to_work_t);
  if(d->ca_measure)
  {
    _measure_ca(in, roi_in, w2, h2, to_in_t, ca);
    _ca_measured(self, piece, ca);
  }
  else if(ca_on)
  {
    for(int k = 0; k < 4; k++) ca[k] = d->ca[k];
  }
  const float inv_r2 = 1.0f / (w2 * w2 + h2 * h2);

  const dt_interpolation_t *interpolation = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);

  DT_OMP_FOR(collapse(2))
  for(int y = 0; y < roi_out->height; y++)
  {
    for(int x = 0; x < roi_out->width; x++)
    {
      const float cx = roi_out->x + x - w2;
      const float cy = roi_out->y + y - h2;
      const float m = t * _ratio(d, tr2 * (cx * cx + cy * cy));
      const float xs = w2 + m * cx;
      const float ys = h2 + m * cy;
      float *const o = out + 4 * ((size_t)y * roi_out->width + x);

      // outside the input image, as uncovered corners are with
      // auto-scale off
      if(xs < -0.5f || ys < -0.5f || xs > orig_w - 0.5f || ys > orig_h - 0.5f)
      {
        for_four_channels(c) o[c] = 0.0f;
        continue;
      }

      const float xi = CLAMP(xs - roi_in->x, 0.0f, limw);
      const float yi = CLAMP(ys - roi_in->y, 0.0f, limh);
      dt_interpolation_compute_pixel4c(interpolation, in, o, xi, yi,
                                       roi_in->width, roi_in->height, 4 * roi_in->width);

      if(ca_on)
      {
        // green decides whether a pixel is inside the image; red and
        // blue are clamped, so the corners do not get colored edges
        float sr, sb;
        _ca_scales(ca, m * m * (cx * cx + cy * cy) * inv_r2, &sr, &sb);
        dt_aligned_pixel_t pr, pg, pb, ir, ig, ib;
        dt_interpolation_compute_pixel4c(interpolation, in, pr,
                                         CLAMP(w2 + sr * m * cx - roi_in->x, 0.0f, limw),
                                         CLAMP(h2 + sr * m * cy - roi_in->y, 0.0f, limh),
                                         roi_in->width, roi_in->height, 4 * roi_in->width);
        dt_interpolation_compute_pixel4c(interpolation, in, pb,
                                         CLAMP(w2 + sb * m * cx - roi_in->x, 0.0f, limw),
                                         CLAMP(h2 + sb * m * cy - roi_in->y, 0.0f, limh),
                                         roi_in->width, roi_in->height, 4 * roi_in->width);
        for_four_channels(c) pg[c] = o[c];

        // each channel of the input profile comes from its own position
        dt_apply_transposed_color_matrix(pr, to_in_t, ir);
        dt_apply_transposed_color_matrix(pg, to_in_t, ig);
        dt_apply_transposed_color_matrix(pb, to_in_t, ib);
        const dt_aligned_pixel_t mixed = { ir[0], ig[1], ib[2], 0.0f };
        dt_aligned_pixel_t res;
        dt_apply_transposed_color_matrix(mixed, to_work_t, res);
        for_each_channel(c) o[c] = res[c];
      }
    }
  }
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_iop_distortion_data_t *d = piece->data;
  const dt_iop_distortion_global_data_t *gd = self->global_data;

  const float orig_w = roi_in->scale * piece->buf_in.width;
  const float orig_h = roi_in->scale * piece->buf_in.height;
  const float w2 = 0.5f * orig_w;
  const float h2 = 0.5f * orig_h;
  const float t = _get_scale(d, piece->buf_in.width, piece->buf_in.height);
  const float tr = t / dt_fast_hypotf(w2, h2);

  const int devid = piece->pipe->devid;
  const int ca_on = d->ca_correct;
  float ca[4] = { 0.0f };
  dt_colormatrix_t to_in_t, to_work_t;
  _ca_space(self, piece, to_in_t, to_work_t);
  if(d->ca_measure)
  {
    float *host = dt_alloc_align_float((size_t)4 * roi_in->width * roi_in->height);
    if(!host) return DT_OPENCL_SYSMEM_ALLOCATION;
    const cl_int err = dt_opencl_copy_image_to_host(devid, host, dev_in,
                                                    roi_in->width, roi_in->height,
                                                    4 * sizeof(float));
    if(err == CL_SUCCESS)
    {
      _measure_ca(host, roi_in, w2, h2, to_in_t, ca);
      _ca_measured(self, piece, ca);
    }
    dt_free_align(host);
    if(err != CL_SUCCESS) return err;
  }
  else if(ca_on)
  {
    for(int k = 0; k < 4; k++) ca[k] = d->ca[k];
  }
  const float inv_r2 = 1.0f / (w2 * w2 + h2 * h2);

  // to the input profile, then back, row major
  float matrices[18];
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      matrices[3 * i + j] = to_in_t[j][i];
      matrices[9 + 3 * i + j] = to_work_t[j][i];
    }
  cl_mem dev_matrices = dt_opencl_copy_host_to_device_constant(devid, sizeof(matrices), matrices);
  if(!dev_matrices) return DT_OPENCL_DEFAULT_ERROR;

  const dt_interpolation_t *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);
  const int itor_width = itor->width;
  // compute_upsampling_taps() in basic.cl knows lanczos as mode 2 and
  // takes the kernel width separately, so lanczos3 is mode 2 with width 3
  const int itor_mode = itor->id == DT_INTERPOLATION_LANCZOS3 ? DT_INTERPOLATION_LANCZOS2 : itor->id;

  const cl_int err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_manual_distortion,
                                                      roi_out->width, roi_out->height,
                                                      CLARG(dev_in), CLARG(dev_out),
                                                      CLARG(roi_out->width), CLARG(roi_out->height),
                                                      CLARG(roi_in->width), CLARG(roi_in->height),
                                                      CLARG(orig_w), CLARG(orig_h),
                                                      CLARG(w2), CLARG(h2), CLARG(tr), CLARG(t),
                                                      CLARG(d->k1), CLARG(d->k2), CLARG(d->k3),
                                                      CLARG(roi_in->x), CLARG(roi_in->y),
                                                      CLARG(roi_out->x), CLARG(roi_out->y),
                                                      CLARG(itor_mode), CLARG(itor_width),
                                                      CLARG(ca_on), CLARG(inv_r2),
                                                      CLARG(ca[0]), CLARG(ca[1]), CLARG(ca[2]), CLARG(ca[3]),
                                                      CLARG(dev_matrices));
  dt_opencl_release_mem_object(dev_matrices);
  return err;
}
#endif

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  tiling->factor = 2.0f; // in + out
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 4;
  tiling->align = 1;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_distortion_params_t *p = (dt_iop_distortion_params_t *)p1;
  dt_iop_distortion_data_t *d = piece->data;

  d->k1 = DT_DISTORTION_SLIDER_TO_K1 * p->distortion;
  d->k2 = p->k2;
  d->k3 = p->k3;
  d->autoscale = p->autoscale;

  d->ca[0] = p->ca_red_a;
  d->ca[1] = p->ca_red_b;
  d->ca[2] = p->ca_blue_a;
  d->ca[3] = p->ca_blue_b;
  d->ca_measure = p->ca_correct && !p->ca_measured;
  d->ca_correct = d->ca_measure
                  || (p->ca_correct
                      && (d->ca[0] != 0.0f || d->ca[1] != 0.0f
                          || d->ca[2] != 0.0f || d->ca[3] != 0.0f));

  if(d->k1 == 0.0f && d->k2 == 0.0f && d->k3 == 0.0f && !d->ca_correct)
    piece->enabled = FALSE;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_distortion_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 2; // basic.cl, from programs.conf
  dt_iop_distortion_global_data_t *gd = malloc(sizeof(dt_iop_distortion_global_data_t));
  self->data = gd;
  gd->kernel_manual_distortion = dt_opencl_create_kernel(program, "manual_distortion");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_distortion_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_manual_distortion);
  free(self->data);
  self->data = NULL;
}

// stores a measurement of the preview pipe in the history, once
static void _preview_pipe_finished(gpointer instance, dt_iop_module_t *self)
{
  dt_iop_distortion_gui_data_t *g = self->gui_data;
  dt_iop_distortion_params_t *p = self->params;
  if(!g) return;

  float ca[4];
  dt_iop_gui_enter_critical_section(self);
  const gboolean ready = g->ca_ready && g->ca_imgid == self->dev->image_storage.id;
  g->ca_ready = FALSE;
  for(int k = 0; k < 4; k++) ca[k] = g->ca[k];
  dt_iop_gui_leave_critical_section(self);

  if(!ready || !self->enabled || !p->ca_correct || p->ca_measured) return;

  p->ca_red_a = ca[0];
  p->ca_red_b = ca[1];
  p->ca_blue_a = ca[2];
  p->ca_blue_b = ca[3];
  p->ca_measured = TRUE;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_changed(dt_iop_module_t *self, GtkWidget *w, void *previous)
{
  dt_iop_distortion_gui_data_t *g = self->gui_data;
  dt_iop_distortion_params_t *p = self->params;

  // turning the correction on measures it again
  if(w == g->ca_correct && p->ca_correct)
    p->ca_measured = FALSE;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_distortion_gui_data_t *g = IOP_GUI_ALLOC(distortion);

  g->distortion = dt_bauhaus_slider_from_params(self, "distortion");
  gtk_widget_set_tooltip_text(g->distortion,
                              _("positive values correct barrel distortion, where straight lines\n"
                                "bow outward; negative values correct pincushion distortion,\n"
                                "where they bow inward"));

  g->autoscale = dt_bauhaus_toggle_from_params(self, "autoscale");
  gtk_widget_set_tooltip_text(g->autoscale,
                              _("scale the corrected image to fill the frame, so that no empty\n"
                                "areas show and as little of the image as possible is lost"));

  g->ca_correct = dt_bauhaus_toggle_from_params(self, "ca_correct");
  gtk_widget_set_tooltip_text(g->ca_correct,
                              _("automatically correct lateral chromatic aberration, the color\n"
                                "fringes along edges that grow toward the corners. it is measured\n"
                                "from the image when turned on; turn it off and on again to\n"
                                "measure it anew"));

  DT_CONTROL_SIGNAL_HANDLE(DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED, _preview_pipe_finished);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
