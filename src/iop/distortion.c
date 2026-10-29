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
#include "common/interpolation.h"
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
*/

DT_MODULE_INTROSPECTION(1, dt_iop_distortion_params_t)

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
} dt_iop_distortion_params_t;

typedef struct dt_iop_distortion_data_t
{
  float k1, k2, k3;
  gboolean autoscale;
} dt_iop_distortion_data_t;

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
    const float xs = w2 + m * cx;
    const float ys = h2 + m * cy;
    xm = MIN(xm, xs);
    xM = MAX(xM, xs);
    ym = MIN(ym, ys);
    yM = MAX(yM, ys);
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

  const dt_interpolation_t *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);
  const int itor_width = itor->width;
  // compute_upsampling_taps() in basic.cl knows lanczos as mode 2 and
  // takes the kernel width separately, so lanczos3 is mode 2 with width 3
  const int itor_mode = itor->id == DT_INTERPOLATION_LANCZOS3 ? DT_INTERPOLATION_LANCZOS2 : itor->id;

  return dt_opencl_enqueue_kernel_2d_args(piece->pipe->devid, gd->kernel_manual_distortion,
                                          roi_out->width, roi_out->height,
                                          CLARG(dev_in), CLARG(dev_out),
                                          CLARG(roi_out->width), CLARG(roi_out->height),
                                          CLARG(roi_in->width), CLARG(roi_in->height),
                                          CLARG(orig_w), CLARG(orig_h),
                                          CLARG(w2), CLARG(h2), CLARG(tr), CLARG(t),
                                          CLARG(d->k1), CLARG(d->k2), CLARG(d->k3),
                                          CLARG(roi_in->x), CLARG(roi_in->y),
                                          CLARG(roi_out->x), CLARG(roi_out->y),
                                          CLARG(itor_mode), CLARG(itor_width));
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

  if(d->k1 == 0.0f && d->k2 == 0.0f && d->k3 == 0.0f)
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

void gui_init(dt_iop_module_t *self)
{
  GtkWidget *w = dt_bauhaus_slider_from_params(self, "distortion");
  gtk_widget_set_tooltip_text(w, _("positive values correct barrel distortion, where straight lines\n"
                                   "bow outward; negative values correct pincushion distortion,\n"
                                   "where they bow inward"));

  w = dt_bauhaus_toggle_from_params(self, "autoscale");
  gtk_widget_set_tooltip_text(w, _("scale the corrected image to fill the frame, so that no empty\n"
                                   "areas show and as little of the image as possible is lost"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
