# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Mitsuba integrator implementing GNM half-Lambert shading.

This integrator aims to reproduce the shading of render_gnm (which uses
PyRender) so that Mitsuba renders match that pipeline as closely as possible.

The class subclasses `mi.SamplingIntegrator` at module top level, which requires
a Mitsuba variant to already be set. This module is therefore only imported
lazily (after `mi.set_variant(...)`) by the package `__init__.py`'s variant
callback. Callers should invoke the package-level `register()` (see
`__init__.py`) before loading a scene that uses this integrator.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np

# Light intensity empirically chosen to match render_gnm.
_LIGHT_INTENSITY = 3.34

_MITSUBA_VERSION = tuple(
    int(x)
    for x in (
        str(mi.__version__).split('.')  # pyrefly: ignore[missing-attribute]
    )
    if x.isdigit()
)
MITSUBA_USE_LEGACY_BSDF_API = _MITSUBA_VERSION <= (3, 9, 1)
MITSUBA_USE_LEGACY_SAMPLE_API = _MITSUBA_VERSION <= (3, 9, 1)


class GnmHalfLambertIntegrator(mi.SamplingIntegrator):
  """Integrator that implements GNM half-Lambert shading and flat shading."""

  def __init__(self, props: mi.Properties):
    """Initializes the integrator.

    Args:
      props: Mitsuba properties.
    """
    super().__init__(props)
    self.light_dir = mi.ScalarVector3f(props.get('light_dir', [1.0, 1.0, 1.0]))
    self.light_intensity = mi.Float(
        props.get(
            'light_intensity',
            _LIGHT_INTENSITY,
        )
    )
    self.ambient_light = mi.Float(props.get('ambient_light', 0.0))
    self.include_shading = bool(props.get('include_shading', True))
    self.backface_culling = bool(props.get('backface_culling', True))

  def traverse(self, callback: mi.TraversalCallback) -> None:
    """Exposes traversable integrator parameters.

    Args:
      callback: Mitsuba traversal callback.
    """
    super().traverse(callback)
    callback.put('light_dir', self.light_dir, mi.ParamFlags.NonDifferentiable)
    callback.put(
        'light_intensity',
        self.light_intensity,
        mi.ParamFlags.NonDifferentiable,
    )
    callback.put(
        'ambient_light',
        self.ambient_light,
        mi.ParamFlags.NonDifferentiable,
    )

  def sample(
      self,
      scene: mi.Scene,
      sampler: mi.Sampler,
      ray: mi.Ray3f,
      medium: mi.Medium | None = None,
      active: mi.Bool | bool = True,
  ) -> tuple[mi.Spectrum, mi.Bool]:
    """Samples the integrator.

    Args:
      scene: The Mitsuba scene.
      sampler: The Mitsuba sampler.
      ray: The Mitsuba ray.
      medium: The Mitsuba medium.
      active: Whether the ray is active.

    Returns:
      A tuple of the radiance and whether the ray is valid (plus an empty AOV
      list on older Mitsuba versions).
    """
    del sampler, medium
    scene_intersection = scene.ray_intersect(ray, mi.Bool(active))
    is_valid = active & scene_intersection.is_valid()
    if self.backface_culling:
      is_valid &= dr.dot(scene_intersection.n, ray.d) < 0.0

    has_vertex_colors = scene_intersection.shape.has_attribute('vertex_colors')
    vertex_colors = dr.select(
        has_vertex_colors,
        scene_intersection.shape.eval_attribute(
            'vertex_colors', scene_intersection, is_valid & has_vertex_colors
        ),
        mi.Color3f(1.0, 1.0, 1.0),
    )

    if MITSUBA_USE_LEGACY_BSDF_API:
      bsdf = scene_intersection.bsdf()
      diffuse_reflectance = (
          bsdf.eval_diffuse_reflectance(  # pyrefly: ignore[missing-attribute]
              scene_intersection, is_valid
          )
      )
    else:
      diffuse_reflectance = (
          scene_intersection.bsdf()
          .eval_features(scene_intersection, is_valid)
          .diffuse_albedo
      )
    base_color = vertex_colors * diffuse_reflectance

    if self.include_shading:
      radiance = vertex_colors * self._shade_half_lambert(
          scene_intersection=scene_intersection,
          diffuse_reflectance=diffuse_reflectance,
      )
    else:
      radiance = dr.power(
          base_color, 2.2
      )

    radiance = dr.select(is_valid, radiance, mi.Color3f(0.0, 0.0, 0.0))
    if MITSUBA_USE_LEGACY_SAMPLE_API:
      return mi.Spectrum(radiance), is_valid, []  # pyrefly: ignore[bad-return]
    return mi.Spectrum(radiance), is_valid

  def _shade_half_lambert(
      self,
      scene_intersection: mi.SurfaceInteraction3f,
      diffuse_reflectance: mi.Color3f | mi.Spectrum,
  ) -> mi.Color3f:
    """Computes reflected radiance using the GNM half-Lambert shading model.

    Combines a half-Lambert diffuse term with a GLTF metallic-roughness
    specular lobe (Cook-Torrance) so that Mitsuba renders match the shading of
    the render_gnm pipeline.

    The shading math is evaluated in the local Mitsuba shading frame defined by
    `scene_intersection.sh_frame`, following idiomatic Mitsuba BSDF/integrator
    conventions. Both the light and view directions are transformed into that
    orthonormal frame, where the shading normal is (0, 0, 1). Each `N . x` term
    therefore reduces to the z-component of the transformed vector, exposed via
    `mi.Frame3f.cos_theta`. Because `sh_frame.to_local` is an orthonormal
    transform it preserves lengths and dot products, so this formulation is
    numerically equivalent to evaluating the same dot products in world space
    (any difference is at floating-point epsilon from the extra transform).

    Args:
      scene_intersection: Surface interaction at the shading point. Its
        `sh_frame` defines the local shading frame and `wi` is the view
        direction (`-ray.d`) already expressed in that frame.
      diffuse_reflectance: Diffuse reflectance (base color) at the intersection.

    Returns:
      The reflected radiance, before modulation by vertex colors.
    """
    # Transform the (world-space) light direction into the local shading frame;
    # the view direction is already available there as `wi` (== -ray.d).
    light_direction_local = scene_intersection.to_local(
        dr.opaque(mi.Vector3f(self.light_dir))
    )
    view_direction_local = scene_intersection.wi
    half_vector_local = dr.normalize(
        light_direction_local + view_direction_local
    )

    # In the local frame the normal is (0, 0, 1), so N.L and N.V are just the
    # z-components (cos_theta). Half-Lambert-style wrap of N.L (via the 0.8/1.8
    # constants) softens the terminator; n_dot_v and v_dot_h are the usual
    # clamped cosine terms.
    cos_theta_light = mi.Frame3f.cos_theta(light_direction_local)
    cos_theta_view = mi.Frame3f.cos_theta(view_direction_local)
    n_dot_l = dr.clip((cos_theta_light + 0.8) / 1.8, 0.0, 1.0)
    n_dot_v = dr.clip(dr.abs(cos_theta_view), 0.001, 1.0)
    v_dot_h = dr.clip(dr.dot(view_direction_local, half_vector_local), 0.0, 1.0)

    # GLTF PBR specular (Cook-Torrance) with roughness=1.0, metallic=0.0:
    # Schlick Fresnel, Smith geometry term, and a constant (roughness=1) NDF.
    fresnel = 0.04 + mi.Float(0.96) * (
        dr.power(1.0 - v_dot_h, 5.0)
    )
    geometry_view = n_dot_v / (0.5 * n_dot_v + 0.5)
    geometry_light = n_dot_l / (0.5 * n_dot_l + 0.5)
    geometry = geometry_view * geometry_light
    distribution = 1.0 / np.pi

    diffuse_color = mi.Color3f(diffuse_reflectance * 0.96)
    diffuse_contrib = mi.Color3f(
        (1.0 - fresnel)
        * diffuse_color
        / np.pi
    )

    spec_contrib = mi.Color3f(
        (fresnel * geometry * distribution)
        / (
            4.0 * n_dot_l * n_dot_v + 0.001
        )
    )

    directional = (
        n_dot_l * self.light_intensity * (diffuse_contrib + spec_contrib)
    )
    return mi.Color3f(
        (1.0 - self.ambient_light) * directional
        + self.ambient_light * mi.Color3f(diffuse_reflectance)
    )


def register() -> None:
  """Registers the GNM half-Lambert integrator plugin."""
  mi.register_integrator('gnm_half_lambert', GnmHalfLambertIntegrator)
