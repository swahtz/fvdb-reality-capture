# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the composable Gaussian splatting pipeline in ``fvdb_reality_capture.functional``.

The pipeline is checked against :class:`GaussianSplat3d`, which composes the same stages, and
against itself across the dense, sparse, cropped and world-space paths.
"""

import unittest
import weakref
from dataclasses import astuple
from unittest import mock

import numpy as np
import torch
from fvdb import JaggedTensor
from fvdb.utils.tests import get_fvdb_test_data_path

import fvdb_reality_capture.functional as F
from fvdb_reality_capture import CameraModel, GaussianRenderMode, GaussianSplat3d, ProjectionMethod
from fvdb_reality_capture.radiance_fields._gaussian_rendering import (
    ImageSpaceRenderBackend,
    RoutedRenderBackend,
    WorldSpaceRenderBackend,
    _ProjectedTrainingView,
    _WorldSpaceTrainingView,
    make_render_backend,
)
from fvdb_reality_capture.radiance_fields._private.utils import crop_image_batch, crop_loss_weight
from fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction import (
    GaussianSplatReconstructionConfig,
    _check_crop_size,
    _DepthTargets,
    _train_crop,
)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class FunctionalPipelineTestCase(unittest.TestCase):
    """Load the garden scene once per test and expose its tensors."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Gaussian splatting requires a CUDA device")
        torch.manual_seed(0)
        np.random.seed(0)
        self.device = torch.device("cuda:0")
        data = np.load(get_fvdb_test_data_path() / "gsplat" / "test_garden_cropped.npz")

        self.means = torch.from_numpy(data["means3d"]).float().to(self.device)
        self.quats = torch.from_numpy(data["quats"]).float().to(self.device)
        self.log_scales = torch.log(torch.from_numpy(data["scales"]).float().to(self.device))
        self.logit_opacities = torch.logit(torch.from_numpy(data["opacities"]).float().to(self.device))
        colors = torch.from_numpy(data["colors"]).float().to(self.device)
        self.W = int(data["width"].item())
        self.H = int(data["height"].item())
        all_w2c = torch.from_numpy(data["viewmats"]).float().to(self.device)
        all_K = torch.from_numpy(data["Ks"]).float().to(self.device)
        self.w2c = all_w2c[:2].contiguous()
        self.K = all_K[:2].contiguous()
        self.C = self.w2c.shape[0]

        N = self.means.shape[0]
        self.sh_degree = 2
        sh = torch.zeros(N, (self.sh_degree + 1) ** 2, 3, device=self.device)
        sh[:, 0] = rgb_to_sh(colors)
        sh[:, 1:] = torch.randn_like(sh[:, 1:]) * 0.05
        self.sh0 = sh[:, :1].contiguous()
        self.shN = sh[:, 1:].contiguous()

    def _params(self, requires_grad: bool = False):
        tensors = [self.means, self.quats, self.log_scales, self.logit_opacities, self.sh0, self.shN]
        return [t.detach().clone().requires_grad_(requires_grad) for t in tensors]

    def _model(self, params) -> GaussianSplat3d:
        means, quats, log_scales, logit_opacities, sh0, shN = params
        return GaussianSplat3d.from_tensors(
            means=means, quats=quats, log_scales=log_scales, logit_opacities=logit_opacities, sh0=sh0, shN=shN
        )

    def _render_functional(self, params, render_mode=GaussianRenderMode.FEATURES, **kwargs):
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H, **kwargs)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=render_mode)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        return F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)


class TestStageOutputs(FunctionalPipelineTestCase):
    def test_projection_contract(self):
        params = self._params()
        projected = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H, antialias=True)
        C, N = self.C, self.means.shape[0]
        self.assertEqual(tuple(projected.radii.shape), (C, N, 2))
        self.assertEqual(tuple(projected.means2d.shape), (C, N, 2))
        self.assertEqual(tuple(projected.depths.shape), (C, N))
        self.assertEqual(tuple(projected.conics.shape), (C, N, 3))
        self.assertEqual(tuple(projected.compensations.shape), (C, N))
        self.assertEqual((projected.num_cameras, projected.num_gaussians), (C, N))
        self.assertEqual(projected.projection_method, ProjectionMethod.ANALYTIC)
        self.assertTrue(projected.is_differentiable)
        with self.assertRaises(Exception):
            projected.radii = None  # frozen
        # Equality is identity, so the dataclasses can be compared and hashed despite holding tensors.
        self.assertEqual(projected, projected)
        self.assertNotEqual(projected, without := F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H))
        self.assertEqual(len({projected, without}), 2)

        without = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H, antialias=False)
        self.assertIsNone(without.compensations)

    def test_render_modes(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        depth = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        both = F.evaluate_gaussian_sh(
            means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH
        )
        self.assertEqual(features.shape[-1], 3)
        self.assertEqual(depth.shape[-1], 1)
        self.assertEqual(both.shape[-1], 4)
        torch.testing.assert_close(both[..., :3], features)
        torch.testing.assert_close(both[..., 3:], depth)
        # The projection zeroes the depth of Gaussians it culls; the features stage gives their true depth.
        # Only visible Gaussians are rasterized, so the two agree where it matters.
        visible = (projected.radii > 0).all(-1)
        torch.testing.assert_close(depth[..., 0][visible], projected.depths[visible])
        self.assertTrue(bool((projected.depths[~visible] == 0).all()))
        # Lower degrees are allowed, higher than available are not.
        F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, sh_degree_to_use=0)
        with self.assertRaises(ValueError):
            F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, sh_degree_to_use=self.sh_degree + 1)

    def test_tile_intersection_contract(self):
        params = self._params()
        projected = F.project_gaussians(*params[:3], self.w2c, self.K, self.W, self.H)
        tiles = F.intersect_gaussian_tiles(projected, F.compute_gaussian_opacities(params[3], projected), tile_size=16)
        self.assertEqual(tuple(tiles.tile_offsets.shape), (self.C, tiles.num_tiles_h, tiles.num_tiles_w))
        self.assertEqual(tiles.num_tiles_h, -(-self.H // 16))
        self.assertEqual(tiles.num_tiles_w, -(-self.W // 16))
        # Culling with opacities can only remove intersections relative to the bounding-box test.
        loose = F.intersect_gaussian_tiles(projected, None, tile_size=16)
        self.assertLessEqual(tiles.tile_gaussian_ids.numel(), loose.tile_gaussian_ids.numel())


class TestMatchesGaussianSplat3d(FunctionalPipelineTestCase):
    def test_forward_matches_oo(self):
        params = self._params()
        images, alphas = self._render_functional(params)
        images_oo, alphas_oo = self._model(params).render_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        torch.testing.assert_close(images, images_oo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas, alphas_oo, atol=1e-5, rtol=1e-5)

    def test_backward_matches_oo(self):
        params_fn = self._params(requires_grad=True)
        params_oo = self._params(requires_grad=True)
        images, alphas = self._render_functional(params_fn)
        (images.square().mean() + alphas.mean()).backward()
        images_oo, alphas_oo = self._model(params_oo).render_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        (images_oo.square().mean() + alphas_oo.mean()).backward()
        for p_fn, p_oo in zip(params_fn, params_oo):
            self.assertIsNotNone(p_fn.grad)
            self.assertGreater(float(p_fn.grad.abs().max()), 0.0)
            # Both paths run the same atomicAdd kernels, so agreement is up to summation order.
            torch.testing.assert_close(p_fn.grad, p_oo.grad, atol=1e-5, rtol=1e-4)

    def test_depth_and_features_and_depth_match_oo(self):
        params = self._params()
        model = self._model(params)
        depth, alpha = self._render_functional(params, render_mode=GaussianRenderMode.DEPTH)
        depth_oo, alpha_oo = model.render_depths(self.w2c, self.K, self.W, self.H, 0.01, 1e10, min_radius_2d=0.0)
        torch.testing.assert_close(depth, depth_oo, atol=1e-5, rtol=1e-5)
        both, _ = self._render_functional(params, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH)
        both_oo, _ = model.render_images_and_depths(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        torch.testing.assert_close(both, both_oo, atol=1e-5, rtol=1e-5)

    def test_world_space_matches_oo_and_is_differentiable_through_ut(self):
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, self.W, self.H, projection_method=ProjectionMethod.UNSCENTED
        )
        self.assertFalse(projected.is_differentiable)
        self.assertFalse(projected.means2d.requires_grad)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        images, alphas = F.rasterize_world_space_gaussians(
            means, quats, log_scales, projected, features, opacities, self.w2c, self.K, tiles
        )
        images.mean().backward()
        for p in (means, quats, log_scales, logit_opacities, sh0):
            self.assertGreater(float(p.grad.abs().max()), 0.0)

        # OpenCV models need explicit distortion coefficients; zeros would silently render pinhole rays.
        opencv = F.project_gaussians(
            means.detach(),
            quats.detach(),
            log_scales.detach(),
            self.w2c,
            self.K,
            self.W,
            self.H,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=torch.zeros(self.C, 12, device=self.device),
        )
        with self.assertRaises(RuntimeError):
            F.rasterize_world_space_gaussians(
                means, quats, log_scales, opencv, features.detach(), opacities.detach(), self.w2c, self.K, tiles
            )
        # The coefficients are checked before the kernel reads twelve per camera.
        world_args = (means, quats, log_scales, opencv, features.detach(), opacities.detach(), self.w2c, self.K, tiles)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            F.rasterize_world_space_gaussians(*world_args, distortion_coeffs=torch.zeros(self.C, 5, device=self.device))
        with self.assertRaisesRegex(RuntimeError, "must be on"):
            F.rasterize_world_space_gaussians(*world_args, distortion_coeffs=torch.zeros(self.C, 12))
        # Pinhole cameras ignore the coefficients, whatever is passed.
        F.rasterize_world_space_gaussians(
            means,
            quats,
            log_scales,
            projected,
            features.detach(),
            opacities.detach(),
            self.w2c,
            self.K,
            tiles,
            distortion_coeffs=torch.zeros(self.C, 5),
        )

        params_oo = self._params()
        images_oo, alphas_oo = self._model(params_oo).render_images_from_world(
            self.w2c, self.K, self.W, self.H, 0.01, 1e10, projection_method=ProjectionMethod.UNSCENTED
        )
        torch.testing.assert_close(images.detach(), images_oo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas.detach(), alphas_oo, atol=1e-5, rtol=1e-5)

    def test_depth_is_differentiable_through_the_unscented_projection(self):
        means, quats, log_scales, logit_opacities, sh0, shN = self._params(requires_grad=True)
        analytic = F.project_gaussians(
            means.detach(), quats.detach(), log_scales.detach(), self.w2c, self.K, self.W, self.H
        )
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, self.W, self.H, projection_method=ProjectionMethod.UNSCENTED
        )
        self.assertFalse(projected.depths.requires_grad)
        # Depth is linear in the center, so the recomputed depth matches the projection's exactly and
        # carries the gradient the unscented kernel cannot provide.
        depth = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        self.assertTrue(depth.requires_grad)
        # Each projection zeroes the depth of what it culls, and the two cull slightly differently.
        visible = (analytic.radii.amin(-1) > 0) & (projected.radii.amin(-1) > 0)
        torch.testing.assert_close(depth[..., 0][visible], analytic.depths[visible], atol=1e-4, rtol=1e-4)
        depth.sum().backward()
        self.assertGreater(float(means.grad.abs().max()), 0.0)
        both = F.evaluate_gaussian_sh(
            means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.FEATURES_AND_DEPTH
        )
        self.assertTrue(both[..., 3:].requires_grad)
        # The depth channel is the same function of its inputs with or without autograd.
        with torch.no_grad():
            same = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected, render_mode=GaussianRenderMode.DEPTH)
        self.assertTrue(torch.equal(same, both[..., 3:].detach()))
        visible = (projected.radii > 0).all(-1)
        torch.testing.assert_close(same[..., 0][visible], projected.depths[visible])
        self.assertFalse(F.requires_distortion_coeffs(CameraModel.PINHOLE))
        self.assertTrue(F.requires_distortion_coeffs(CameraModel.OPENCV_RADTAN_5))

    def test_training_loop_reduces_loss(self):
        target, _ = self._render_functional(self._params())
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        with torch.no_grad():
            sh0.add_(0.3 * torch.randn_like(sh0))
        optimizer = torch.optim.Adam([sh0, shN, logit_opacities], lr=1e-2)
        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            images, _ = self._render_functional(params)
            loss = torch.nn.functional.l1_loss(images, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.5 * losses[0])


class TestSparseAndCrop(FunctionalPipelineTestCase):
    def _pixels(self, with_duplicates: bool) -> JaggedTensor:
        rows = torch.randint(0, self.H, (200,), device=self.device)
        cols = torch.randint(0, self.W, (200,), device=self.device)
        px = torch.stack([rows, cols], dim=1)
        px = torch.unique(px, dim=0)
        if with_duplicates:
            px = torch.cat([px, px[:37], px[5:6].repeat(3, 1)])
        return JaggedTensor([px, px.flip(0)])

    def test_sparse_matches_dense_at_requested_pixels(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        dense, dense_alpha = F.rasterize_screen_space_gaussians(
            projected, features, opacities, F.intersect_gaussian_tiles(projected, opacities)
        )
        for with_duplicates in (False, True):
            pixels = self._pixels(with_duplicates)
            sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
            self.assertEqual(sparse_tiles.has_duplicates, with_duplicates)
            rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
            self.assertEqual(len(rendered), self.C)
            all_tiles = torch.ones_like(sparse_tiles.active_tile_mask)
            same, _ = F.rasterize_screen_space_gaussians_sparse(
                projected, features, opacities, sparse_tiles, tile_masks=all_tiles
            )
            torch.testing.assert_close(same.jdata, rendered.jdata)
            # A per-pixel mask handed in as a tile mask is refused instead of skipping the wrong tiles.
            with self.assertRaisesRegex(ValueError, "per-tile"):
                F.rasterize_screen_space_gaussians_sparse(
                    projected, features, opacities, sparse_tiles, tile_masks=torch.ones(self.C, self.H, self.W)
                )
            for c in range(self.C):
                px = pixels[c].jdata
                self.assertEqual(tuple(rendered[c].jdata.shape), (px.shape[0], 3))
                torch.testing.assert_close(rendered[c].jdata, dense[c, px[:, 0], px[:, 1]], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(alphas[c].jdata, dense_alpha[c, px[:, 0], px[:, 1]], atol=1e-5, rtol=1e-5)

    def test_sparse_matches_oo_and_backward_runs(self):
        params = self._params(requires_grad=True)
        means, quats, log_scales, logit_opacities, sh0, shN = params
        pixels = self._pixels(with_duplicates=True)
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
        rendered_oo, alphas_oo = self._model(self._params()).sparse_render_images(
            pixels, self.w2c, self.K, self.W, self.H, 0.01, 1e10
        )
        torch.testing.assert_close(rendered.jdata.detach(), rendered_oo.jdata, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(alphas.jdata.detach(), alphas_oo.jdata, atol=1e-5, rtol=1e-5)
        rendered.jdata.mean().backward()
        self.assertGreater(float(means.grad.abs().max()), 0.0)
        self.assertGreater(float(sh0.grad.abs().max()), 0.0)

    def test_crop_is_a_slice_of_the_full_render(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        full, full_alpha = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)
        ox, oy, w, h = 37, 21, 90, 60
        crop, crop_alpha = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, crop=(ox, oy, w, h)
        )
        self.assertEqual(tuple(crop.shape), (self.C, h, w, 3))
        torch.testing.assert_close(crop, full[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(crop_alpha, full_alpha[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)
        # Clamped to the image, invalid crops rejected.
        clamped, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, crop=(self.W - 10, self.H - 5, 100, 100)
        )
        self.assertEqual(tuple(clamped.shape[1:3]), (5, 10))
        for bad in ((-1, 0, 10, 10), (0, 0, 0, 10)):
            with self.assertRaises(ValueError):
                F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, crop=bad)
        # A crop entirely outside the image clips to nothing rather than raising.
        empty, _ = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, crop=(self.W, 0, 10, 10))
        self.assertEqual(tuple(empty.shape), (self.C, 0, 0, 3))

        # The OO crop path agrees, including a mask given in crop coordinates.
        model = self._model(params)
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        crop_oo, _ = model.render_from_projected_gaussians(
            pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy
        )
        torch.testing.assert_close(crop_oo, crop, atol=1e-5, rtol=1e-5)
        mask = torch.zeros(self.C, h, w, dtype=torch.bool, device=self.device)
        mask[:, : h // 2] = True
        masked, masked_alpha = model.render_from_projected_gaussians(
            pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=mask
        )
        torch.testing.assert_close(masked[:, : h // 2], crop[:, : h // 2], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(masked[:, h // 2 :].abs().max()), 0.0)
        self.assertEqual(float(masked_alpha[:, h // 2 :].abs().max()), 0.0)
        # A crop past the image edge keeps its requested size: the part inside the image is the render,
        # the rest is background with zero alpha. A crop-space mask of the requested size still applies.
        edge_w, edge_h = 50, 40
        edge_mask = torch.ones(self.C, edge_h, edge_w, dtype=torch.bool, device=self.device)
        edge, edge_alpha = model.render_from_projected_gaussians(
            pg,
            crop_width=edge_w,
            crop_height=edge_h,
            crop_origin_w=self.W - 30,
            crop_origin_h=self.H - 25,
            masks=edge_mask,
        )
        self.assertEqual(tuple(edge.shape[1:3]), (edge_h, edge_w))
        self.assertEqual(tuple(edge_alpha.shape[1:]), (edge_h, edge_w, 1))
        torch.testing.assert_close(edge[:, :25, :30], full[:, self.H - 25 :, self.W - 30 :], atol=1e-5, rtol=1e-5)
        self.assertEqual(float(edge[:, 25:].abs().max()), 0.0)
        self.assertEqual(float(edge[:, :, 30:].abs().max()), 0.0)
        self.assertEqual(float(edge_alpha[:, 25:].abs().max()), 0.0)
        self.assertEqual(float(edge_alpha[:, :, 30:].abs().max()), 0.0)
        # Non-boolean masks are accepted with and without a crop.
        float_mask = torch.ones(self.C, self.H, self.W, device=self.device)
        with_float, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, masks=float_mask, crop=(ox, oy, w, h)
        )
        torch.testing.assert_close(with_float, crop, atol=1e-5, rtol=1e-5)
        # With a crop the stage also takes a crop-sized mask, and pools the skipped tiles from it.
        stage_masked, _ = F.rasterize_screen_space_gaussians(
            projected, features, opacities, tiles, masks=mask, crop=(ox, oy, w, h)
        )
        torch.testing.assert_close(stage_masked, masked, atol=1e-5, rtol=1e-5)
        # A mask of any other size is rejected rather than read from its top-left corner; the class
        # method takes the requested or the clipped crop size only.
        with self.assertRaisesRegex(ValueError, "full-image mask .* or a crop mask"):
            F.rasterize_screen_space_gaussians(
                projected, features, opacities, tiles, masks=mask[:, :-1], crop=(ox, oy, w, h)
            )
        with self.assertRaisesRegex(ValueError, "match the crop"):
            model.render_from_projected_gaussians(
                pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=float_mask
            )
        # A mask on the wrong device is rejected at the same point, not deep inside torch.
        cpu_full_mask = torch.ones(self.C, self.H, self.W, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "must be on"):
            F.rasterize_screen_space_gaussians(projected, features, opacities, tiles, masks=cpu_full_mask)
        with self.assertRaisesRegex(ValueError, "must be on"):
            model.render_from_projected_gaussians(
                pg, crop_width=w, crop_height=h, crop_origin_w=ox, crop_origin_h=oy, masks=mask.cpu()
            )

    def test_analysis_matches_oo(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        model = self._model(params)
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        counts, alphas = F.rasterize_num_contributing_gaussians(projected, opacities, tiles)
        counts_oo, alphas_oo = model.render_num_contributing_gaussians(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertTrue(torch.equal(counts, counts_oo))
        torch.testing.assert_close(alphas, alphas_oo)

        ids, weights = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles)
        ids_oo, weights_oo = model.render_contributing_gaussian_ids(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertTrue(torch.equal(ids.jdata, ids_oo.jdata))
        self.assertEqual(ids.ldim, 2)
        self.assertEqual(int(ids.jdata.numel()), int(counts.sum()))
        top_ids, _ = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles, top_k_contributors=3)
        self.assertLessEqual(int(top_ids.jdata.numel()), int(counts.clamp(max=3).sum()))

    def test_sparse_analysis_with_duplicates(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        counts_dense, _ = F.rasterize_num_contributing_gaussians(projected, opacities, tiles)
        ids_dense, _ = F.rasterize_contributing_gaussian_ids(projected, opacities, tiles)

        pixels = self._pixels(with_duplicates=True)
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        ids, weights = F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(ids.ldim, 2)
        for c in range(self.C):
            px = pixels[c].jdata
            self.assertTrue(torch.equal(counts[c].jdata, counts_dense[c, px[:, 0], px[:, 1]]))

            def flat(x):
                return x.jdata if isinstance(x, JaggedTensor) else x

            expected = [flat(ids_dense[c][int(r) * self.W + int(q)]) for r, q in px.tolist()]
            got = [flat(t) for t in ids[c].unbind()]
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                self.assertTrue(torch.equal(g, e))

    def test_sparse_analysis_keeps_a_trailing_camera_with_no_requested_pixels(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        w2c, K = self.w2c[:2], self.K[:2]
        projected = F.project_gaussians(means, quats, log_scales, w2c, K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        busy = torch.tensor([[self.H // 2, self.W // 2], [self.H // 2 + 3, self.W // 2 + 3]], device=self.device)
        empty = torch.empty(0, 2, dtype=torch.int64, device=self.device)
        # Camera 0 has a duplicate (so the expansion path runs); camera 1 requests nothing.
        pixels = JaggedTensor([torch.cat([busy, busy[:1]]), empty])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        self.assertTrue(sparse_tiles.has_duplicates)
        for result in F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles):
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result[0].unbind()), 3)
            self.assertEqual(len(result[1].unbind()), 0)
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(len(counts), 2)
        self.assertEqual(counts[1].jdata.numel(), 0)

    def test_single_camera_sparse_analysis_keeps_camera_nesting(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        w2c, K = self.w2c[:1], self.K[:1]
        projected = F.project_gaussians(means, quats, log_scales, w2c, K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        pixels = self._pixels(with_duplicates=True)[0]  # one camera, duplicates included
        pixels = JaggedTensor([pixels.jdata])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(pixels, projected, opacities)
        ids, weights = F.rasterize_contributing_gaussian_ids_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(ids.ldim, 2)
        self.assertEqual(len(ids), 1)
        self.assertEqual(len(ids[0].unbind()), pixels.jdata.shape[0])
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(int(ids.jdata.numel()), int(counts.jdata.sum()))

    def test_opacities_are_validated_and_made_contiguous(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(
            means, quats, log_scales, self.w2c, self.K, image_width=self.W, image_height=self.H
        )
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        self.assertEqual(tuple(opacities.shape), (self.C, means.shape[0]))
        tiles = F.intersect_gaussian_tiles(projected, opacities, tile_size=16)
        a, _ = F.rasterize_screen_space_gaussians(projected, features, opacities, tiles)
        # Every stage checks the opacities against the projection's camera and Gaussian counts.
        with self.assertRaises(ValueError):
            F.rasterize_screen_space_gaussians(projected, features, opacities[:1], tiles)
        with self.assertRaises(ValueError):
            F.intersect_gaussian_tiles(projected, opacities[:, :-1], tile_size=16)
        with self.assertRaises(ValueError):
            F.rasterize_num_contributing_gaussians(projected, torch.sigmoid(logit_opacities), tiles)
        # A non-contiguous (expanded) opacity tensor is accepted and materialized, not handed to the kernel.
        expanded = torch.sigmoid(logit_opacities).unsqueeze(0).expand(self.C, -1)
        self.assertFalse(expanded.is_contiguous())
        c, _ = F.rasterize_screen_space_gaussians(projected, features, expanded, tiles)
        torch.testing.assert_close(c, a)

    def test_projected_splats_snapshot_opacities_at_projection(self):
        model = self._model(self._params(requires_grad=True))
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        # Opacities and features come from the same projection, so they carry a gradient together.
        self.assertTrue(pg.opacities.requires_grad)
        self.assertTrue(pg.render_quantities.requires_grad)
        expected = torch.sigmoid(model.logit_opacities.detach()).repeat(self.C, 1)
        torch.testing.assert_close(pg.opacities.detach(), expected)
        # An optimizer step between projection and render must not leak into the projection.
        with torch.no_grad():
            model.logit_opacities.add_(1.0)
        torch.testing.assert_close(pg.opacities.detach(), expected)
        self.assertIs(pg.opacities, pg.opacities)
        # A projection made without a graph renders without one, for both quantities.
        with torch.no_grad():
            preview = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertFalse(preview.opacities.requires_grad)
        self.assertFalse(preview.render_quantities.requires_grad)

    def test_deduplicate_pixels_keeps_request_order_and_never_merges_out_of_image_pixels(self):
        w, h = 64, 48
        # (0, 64) linearizes to the same key as (1, 0) in a 64-wide image; it must not be merged into it.
        coords = torch.tensor([[3, 7], [1, 2], [3, 7], [0, 64], [1, 0], [5, 5], [1, 2]], dtype=torch.int64)
        coords = coords.to(self.device)
        unique, inverse, has_duplicates = F.deduplicate_pixels(JaggedTensor([coords]), w, h)
        self.assertTrue(has_duplicates)
        self.assertTrue(torch.equal(unique.jdata, coords[[0, 1, 3, 4, 5]]), "first occurrences, in request order")
        self.assertTrue(torch.equal(unique.jdata[inverse], coords))

    def test_as_pixel_jagged_validates_shape_and_dtype(self):
        with self.assertRaises(TypeError):
            F.as_pixel_jagged(torch.zeros(1, 4, 2, device=self.device))
        with self.assertRaises(ValueError):
            F.as_pixel_jagged(torch.zeros(0, 4, 2, dtype=torch.int64, device=self.device))
        with self.assertRaises(ValueError):
            F.as_pixel_jagged(JaggedTensor([torch.zeros(4, 3, dtype=torch.int64, device=self.device)]))
        pixels = F.as_pixel_jagged(torch.zeros(self.C, 4, 2, dtype=torch.int32, device=self.device))
        self.assertEqual(pixels.num_tensors, self.C)

    def test_projected_splats_render_crops_outside_the_image_as_background(self):
        model = self._model(self._params())
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        self.assertIs(pg.opacities, pg.opacities)
        # Tile intersections are computed on demand, not held by the projection.
        self.assertIsNot(pg.tile_intersection(16), pg.tile_intersection(16))
        background = torch.tensor([[0.2, 0.4, 0.6]] * self.C, device=self.device)
        for origin_w, origin_h in ((self.W, 0), (0, self.H), (self.W + 5, self.H + 5)):
            images, alphas = model.render_from_projected_gaussians(
                pg,
                crop_width=10,
                crop_height=10,
                crop_origin_w=origin_w,
                crop_origin_h=origin_h,
                backgrounds=background,
            )
            self.assertEqual(tuple(images.shape), (self.C, 10, 10, 3))
            torch.testing.assert_close(images, background[:, None, None, :].expand_as(images))
            self.assertEqual(float(alphas.abs().max()), 0.0)

    def test_crops_outside_the_image_stay_differentiable(self):
        params = self._params(requires_grad=True)
        model = self._model(params)
        outside = dict(crop_width=10, crop_height=10, crop_origin_w=self.W, crop_origin_h=0)
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        images, alphas = model.render_from_projected_gaussians(pg, **outside)
        self.assertTrue(images.requires_grad)
        self.assertTrue(alphas.requires_grad)
        # Nothing was rendered, so backward runs and every gradient it produces is zero.
        (images.sum() + alphas.sum()).backward()
        grads = [p.grad for p in params if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(float(g.abs().max()) == 0.0 for g in grads))
        with torch.no_grad():
            pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
            images, alphas = model.render_from_projected_gaussians(pg, **outside)
        self.assertFalse(images.requires_grad)
        self.assertFalse(alphas.requires_grad)
        # The stage function has the same contract minus the padding: an all-outside crop is empty.
        empty, empty_alpha = F.rasterize_screen_space_gaussians(
            pg.projected_gaussians,
            pg.render_quantities,
            pg.opacities,
            pg.tile_intersection(16),
            crop=(self.W, 0, 10, 10),
        )
        self.assertEqual(tuple(empty.shape), (self.C, 0, 0, 3))
        self.assertEqual(tuple(empty_alpha.shape), (self.C, 0, 0, 1))
        # A mask of the wrong shape is rejected for an outside crop just as for any other crop.
        bad_mask = torch.ones(self.C, 3, 3, dtype=torch.bool, device=self.device)
        with self.assertRaisesRegex(ValueError, "masks must match"):
            model.render_from_projected_gaussians(pg, masks=bad_mask, **outside)
        good_mask = torch.ones(self.C, 10, 10, dtype=torch.bool, device=self.device)
        images, _ = model.render_from_projected_gaussians(pg, masks=good_mask, **outside)
        self.assertEqual(tuple(images.shape), (self.C, 10, 10, 3))

    def test_empty_selection(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        empty = JaggedTensor([torch.empty(0, 2, dtype=torch.int64, device=self.device) for _ in range(self.C)])
        sparse_tiles = F.intersect_gaussian_tiles_sparse(empty, projected, opacities)
        rendered, alphas = F.rasterize_screen_space_gaussians_sparse(projected, features, opacities, sparse_tiles)
        self.assertEqual(tuple(rendered.jdata.shape), (0, 3))
        self.assertEqual(tuple(alphas.jdata.shape), (0, 1))
        counts, _ = F.rasterize_num_contributing_gaussians_sparse(projected, opacities, sparse_tiles)
        self.assertEqual(counts.jdata.numel(), 0)


class TestRenderBackends(FunctionalPipelineTestCase):
    """The training backends do the crop-independent work once per view."""

    def _camera_batch(self, camera_model: CameraModel = CameraModel.PINHOLE):
        camera_models = torch.full((self.C,), int(camera_model), dtype=torch.int32)
        distortion_coeffs = torch.zeros(self.C, 12, device=self.device)
        return camera_models, distortion_coeffs

    def _forward_train(self, backend, model, config, camera_model: CameraModel = CameraModel.PINHOLE):
        camera_models, distortion_coeffs = self._camera_batch(camera_model)
        return backend.forward_train(
            model=model,
            config=config,
            world_to_camera_matrices=self.w2c,
            projection_matrices=self.K,
            camera_models=camera_models,
            distortion_coeffs=distortion_coeffs,
            image_width=self.W,
            image_height=self.H,
            sh_degree_to_use=self.sh_degree,
        )

    def _assert_crops_are_slices(self, view):
        full = view.render_crop((0, 0, self.W, self.H))
        self.assertEqual(tuple(full.image.shape), (self.C, self.H, self.W, 3))
        gt = torch.zeros(self.C, self.H, self.W, 3)
        for _, _, crop, _ in crop_image_batch(gt, None, 2):
            ox, oy, w, h = crop
            out = view.render_crop(crop)
            torch.testing.assert_close(out.image, full.image[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out.alpha, full.alpha[:, oy : oy + h, ox : ox + w], atol=1e-5, rtol=1e-5)

    def test_image_space_view_projects_once_and_renders_every_crop_from_it(self):
        model = self._model(self._params())
        with (
            mock.patch.object(
                model, "project_gaussians_for_images", wraps=model.project_gaussians_for_images
            ) as project,
            mock.patch(
                "fvdb_reality_capture.radiance_fields.gaussian_splatting.intersect_gaussian_tiles",
                wraps=F.intersect_gaussian_tiles,
            ) as intersect,
        ):
            view = self._forward_train(ImageSpaceRenderBackend(), model, GaussianSplatReconstructionConfig())
            self._assert_crops_are_slices(view)
        # Projection and tile intersection each ran once for the whole view, however many crops.
        self.assertEqual(project.call_count, 1)
        self.assertEqual(intersect.call_count, 1)
        self.assertIsInstance(view, _ProjectedTrainingView)

    def _train_one_view(self, crops: int) -> tuple[GaussianSplat3d, list[torch.Tensor]]:
        params = self._params(requires_grad=True)
        model = self._model(params)
        model.accumulate_mean_2d_gradients = True
        view = self._forward_train(ImageSpaceRenderBackend(), model, GaussianSplatReconstructionConfig())
        gt = torch.zeros(self.C, self.H, self.W, 3)
        for _, _, crop, _ in crop_image_batch(gt, None, crops):
            out = view.render_crop(crop)
            # Per-pixel mean losses per crop, weighted by the crop's share of the image as the training loop
            # does, so the crops add up to exactly the full-image mean loss. SSIM is left out on purpose: its
            # windows see zero padding at crop seams, so it is not crop-additive.
            weight = crop_loss_weight(crop, self.H, self.W)
            ((out.image.square().mean() + out.alpha.mean()) * weight).backward()
        # Nothing reaches the model until the shared backward runs once.
        self.assertTrue(all(p.grad is None for p in params))
        view.finish_backward()
        return model, params

    def test_crops_share_one_projection_backward_and_one_densification_sample(self):
        whole_model, whole = self._train_one_view(crops=1)
        cropped_model, cropped = self._train_one_view(crops=2)
        for p_whole, p_cropped in zip(whole, cropped):
            self.assertIsNotNone(p_whole.grad)
            # The crop path splits the atomicAdd raster backward into four launches and adds their
            # partial sums, so agreement is up to that summation order (a handful of elements at ~1e-4).
            torch.testing.assert_close(p_cropped.grad, p_whole.grad, atol=1e-3, rtol=1e-3)
        # The projection backward ran once per view in both cases, over the whole image's gradient, so
        # the densification statistics agree. The kernel counts camera-Gaussian pairs, hence <= C.
        self.assertTrue(
            torch.equal(cropped_model.accumulated_gradient_step_counts, whole_model.accumulated_gradient_step_counts)
        )
        self.assertEqual(int(whole_model.accumulated_gradient_step_counts.max()), self.C)
        self.assertGreater(float(whole_model.accumulated_mean_2d_gradient_norms.sum()), 0.0)
        torch.testing.assert_close(
            cropped_model.accumulated_mean_2d_gradient_norms,
            whole_model.accumulated_mean_2d_gradient_norms,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_world_space_view_renders_each_crop_directly(self):
        params = self._params(requires_grad=True)
        model = self._model(params)
        w2c = self.w2c.clone().requires_grad_(True)  # stands in for pose-adjusted cameras
        with mock.patch.object(
            GaussianSplat3d, "project_gaussians_for_images", wraps=model.project_gaussians_for_images
        ) as project:
            camera_models, distortion_coeffs = self._camera_batch(CameraModel.PINHOLE)
            view = WorldSpaceRenderBackend().forward_train(
                model=model,
                config=GaussianSplatReconstructionConfig(),
                world_to_camera_matrices=w2c,
                projection_matrices=self.K,
                camera_models=camera_models,
                distortion_coeffs=distortion_coeffs,
                image_width=self.W,
                image_height=self.H,
                sh_degree_to_use=self.sh_degree,
            )
            self.assertIsInstance(view, _WorldSpaceTrainingView)
            # The projection, features and tiles are computed once; every crop rasterizes from them and
            # equals the slice of the full render.
            self.assertEqual(project.call_count, 1)
            self._assert_crops_are_slices(view)
            self.assertEqual(project.call_count, 1)
            gt = torch.zeros(self.C, self.H, self.W, 3)
            for _, _, crop, _ in crop_image_batch(gt, None, 2):
                view.render_crop(crop).image.sum().backward()
                # The world-space rasterizer reaches the 3D parameters directly, crop by crop.
                self.assertGreater(float(params[0].grad.abs().max()), 0.0)
        # Features and camera matrices are shared by the crops, so their gradients arrive once, at the end.
        self.assertIsNone(w2c.grad)
        self.assertIsNone(params[4].grad)
        view.finish_backward()
        self.assertIsNotNone(w2c.grad)
        self.assertIsNotNone(params[4].grad)

    def test_world_space_training_keeps_the_refinement_accumulators_allocated(self):
        # Refinement reads the radius accumulator on every path, so world space must allocate it (zeroed)
        # even though it wires only the image-space projections into the gradient statistics.
        model = self._model(self._params(requires_grad=True))
        model.accumulate_mean_2d_gradients = True
        model.accumulate_max_2d_radii = True
        view = self._forward_train(WorldSpaceRenderBackend(), model, GaussianSplatReconstructionConfig())
        view.render_crop((0, 0, self.W, self.H)).image.sum().backward()
        view.finish_backward()
        for accumulator in (
            model.accumulated_max_2d_radii,
            model.accumulated_mean_2d_gradient_norms,
            model.accumulated_gradient_step_counts,
        ):
            self.assertIsNotNone(accumulator)
            self.assertEqual(int(accumulator.abs().sum()), 0)

    def test_crops_must_cover_the_ssim_window(self):
        sizes = np.array([[420, 648], [64, 96]])
        _check_crop_size(sizes, 1)
        _check_crop_size(sizes, 5)  # 64 // 5 = 12
        with self.assertRaisesRegex(ValueError, "SSIM window"):
            _check_crop_size(sizes, 6)  # 64 // 6 = 10
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _check_crop_size(sizes, 0)
        # A dataset that delivers patches is judged by the patch size, not the image size.
        with self.assertRaisesRegex(ValueError, "SSIM window"):
            _check_crop_size(sizes, 2, patch_size=20)

    def test_training_a_crop_releases_its_graph_and_the_view_releases_its_copies(self):
        # Image space: the view holds detached copies of the projection and features, plus their gradients.
        params = self._params(requires_grad=True)
        model = self._model(params)
        config = GaussianSplatReconstructionConfig()
        view = self._forward_train(ImageSpaceRenderBackend(), model, config)
        copy = weakref.ref(view._features)
        gt = torch.zeros(self.C, self.H, self.W, 3, dtype=torch.uint8)
        for pixels, mask_pixels, crop, _ in crop_image_batch(gt, None, 2):
            losses = _train_crop(
                view,
                crop,
                crop_loss_weight(crop, self.H, self.W),
                pixels,
                mask_pixels,
                _DepthTargets(),
                config,
                self.device,
            )
            self.assertFalse(any(term.requires_grad for term in astuple(losses)))
        # The crops accumulated into the copy; the shared backward hands that gradient on and drops it.
        self.assertIsNotNone(view._features.grad)
        view.finish_backward()
        self.assertIsNone(view._features.grad)
        self.assertGreater(float(params[0].grad.abs().max()), 0.0)
        # Nothing outside the view holds a crop graph, so the copy dies with the view, before the next step.
        del view
        self.assertIsNone(copy())
        # A loss kept alive past its backward would hold the copy through its AccumulateGrad node.
        view = self._forward_train(ImageSpaceRenderBackend(), model, config)
        copy = weakref.ref(view._features)
        held = view.render_crop((0, 0, self.W, self.H)).image.sum()
        held.backward()
        view.finish_backward()
        del view
        self.assertIsNotNone(copy())
        del held
        self.assertIsNone(copy())

    def test_image_space_backend_renders_forward_only_camera_batches_through_world_space(self):
        # In a scene that mixes pinhole and distortion cameras, the pinhole batches keep image space and
        # keep feeding the densification statistics, while the distortion batches take the world-space
        # path so their geometry still gets a gradient.
        params = self._params(requires_grad=True)
        model = self._model(params)
        model.accumulate_mean_2d_gradients = True
        backend = make_render_backend("auto")
        self.assertIsInstance(backend, RoutedRenderBackend)
        config = GaussianSplatReconstructionConfig()
        full = (0, 0, self.W, self.H)

        pinhole_view = self._forward_train(backend, model, config)
        self.assertIsInstance(pinhole_view, _ProjectedTrainingView)
        pinhole_view.render_crop(full).image.sum().backward()
        pinhole_view.finish_backward()
        norms_after_pinhole = model.accumulated_mean_2d_gradient_norms.clone()
        counts_after_pinhole = model.accumulated_gradient_step_counts.clone()
        self.assertGreater(float(norms_after_pinhole.sum()), 0.0)
        means_grad_after_pinhole = params[0].grad.clone()

        opencv_view = self._forward_train(backend, model, config, camera_model=CameraModel.OPENCV_RADTAN_5)
        self.assertIsInstance(opencv_view, _WorldSpaceTrainingView)
        opencv_view.render_crop(full).image.sum().backward()
        opencv_view.finish_backward()
        self.assertFalse(torch.equal(params[0].grad, means_grad_after_pinhole), "world space trained the geometry")
        # World-space views project without the accumulators, so neither the norms nor the step counts move.
        torch.testing.assert_close(model.accumulated_mean_2d_gradient_norms, norms_after_pinhole)
        self.assertTrue(torch.equal(model.accumulated_gradient_step_counts, counts_after_pinhole))

    def test_routed_validation_reports_forward_only_cameras_and_pure_image_space_rejects_them(self):
        model = self._model(self._params())
        routed = make_render_backend("auto")
        pure = ImageSpaceRenderBackend()
        module_logger = "fvdb_reality_capture.radiance_fields._gaussian_rendering"
        config = GaussianSplatReconstructionConfig()
        opencv = mock.MagicMock(camera_models=np.array([int(CameraModel.OPENCV_RADTAN_5)]), indices=[])
        with self.assertLogs(module_logger, level="WARNING") as logs:
            routed.validate_scene_cameras(model, opencv, config, self.device)
        self.assertIn("OPENCV_RADTAN_5", logs.output[0])
        self.assertIn("world space", logs.output[0])
        self.assertIn("densification", logs.output[0].lower())
        # The default config optimizes poses, and world space gives the camera matrices no gradient.
        self.assertIn("pose optimization", logs.output[0].lower())
        with self.assertLogs(module_logger, level="WARNING") as logs:
            routed.validate_scene_cameras(
                model, opencv, GaussianSplatReconstructionConfig(optimize_camera_poses=False), self.device
            )
        self.assertNotIn("pose optimization", logs.output[0].lower())
        with self.assertLogs(module_logger, level="WARNING") as logs:
            WorldSpaceRenderBackend().validate_scene_cameras(model, opencv, config, self.device)
        self.assertIn("pose optimization", logs.output[0].lower())
        with self.assertRaisesRegex(ValueError, "auto"):
            pure.validate_scene_cameras(model, opencv, config, self.device)
        self.assertIsInstance(make_render_backend("image_space"), ImageSpaceRenderBackend)
        pinhole_unscented = mock.MagicMock(camera_models=np.array([int(CameraModel.PINHOLE)]), indices=[])
        unscented = GaussianSplatReconstructionConfig(projection_method="unscented")
        with self.assertLogs(module_logger, level="WARNING"):
            routed.validate_scene_cameras(model, pinhole_unscented, unscented, self.device)
        with self.assertRaisesRegex(ValueError, "auto"):
            pure.validate_scene_cameras(model, pinhole_unscented, unscented, self.device)
        pinhole = mock.MagicMock(camera_models=np.array([int(CameraModel.PINHOLE)]), indices=[])
        with self.assertNoLogs(module_logger, level="WARNING"):
            routed.validate_scene_cameras(model, pinhole, config, self.device)
            pure.validate_scene_cameras(model, pinhole, config, self.device)

    def test_routed_evaluation_uses_the_renderer_that_trains_each_camera(self):
        model = self._model(self._params())
        routed = make_render_backend("auto")
        config = GaussianSplatReconstructionConfig()
        for camera_model, reference in (
            (CameraModel.OPENCV_RADTAN_5, WorldSpaceRenderBackend()),
            (CameraModel.PINHOLE, ImageSpaceRenderBackend()),
        ):
            camera_models, distortion_coeffs = self._camera_batch(camera_model)
            kwargs = dict(
                model=model,
                config=config,
                world_to_camera_matrices=self.w2c,
                projection_matrices=self.K,
                camera_models=camera_models,
                distortion_coeffs=distortion_coeffs,
                image_width=self.W,
                image_height=self.H,
                sh_degree_to_use=self.sh_degree,
            )
            routed_eval = routed.forward_eval(**kwargs)
            reference_eval = reference.forward_eval(**kwargs)
            torch.testing.assert_close(routed_eval.image, reference_eval.image, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(routed_eval.alpha, reference_eval.alpha, atol=1e-5, rtol=1e-5)

    def test_stale_tiles_are_rejected_and_precomputed_tiles_are_accepted(self):
        params = self._params()
        means, quats, log_scales, logit_opacities, sh0, shN = params
        projected = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        opacities = F.compute_gaussian_opacities(logit_opacities, projected)
        features = F.evaluate_gaussian_sh(means, sh0, shN, self.w2c, projected)
        tiles = F.intersect_gaussian_tiles(projected, opacities)
        # Tiles from a different image size or camera count would index the kernels out of range.
        smaller = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W // 2, self.H // 2)
        smaller_opacities = F.compute_gaussian_opacities(logit_opacities, smaller)
        with self.assertRaisesRegex(ValueError, "intersected for"):
            F.rasterize_screen_space_gaussians(smaller, features, smaller_opacities, tiles)
        with self.assertRaisesRegex(ValueError, "intersected for"):
            F.rasterize_num_contributing_gaussians(smaller, smaller_opacities, tiles)
        # Re-projecting the same cameras (after an optimizer step or a refinement, say) makes new tiles
        # necessary even though the sizes agree, because the ids may point at Gaussians that moved or went.
        again = F.project_gaussians(means, quats, log_scales, self.w2c, self.K, self.W, self.H)
        with self.assertRaisesRegex(ValueError, "different projection"):
            F.rasterize_screen_space_gaussians(again, features, opacities, tiles)
        # Detached copies made with replace() keep the projection's identity, as the training view relies on.
        from dataclasses import replace

        F.rasterize_screen_space_gaussians(
            replace(projected, means2d=projected.means2d.detach()), features, opacities, tiles
        )
        one_camera = F.project_gaussians(means, quats, log_scales, self.w2c[:1], self.K[:1], self.W, self.H)
        with self.assertRaisesRegex(ValueError, "cameras"):
            F.rasterize_screen_space_gaussians(
                one_camera, features[:1], F.compute_gaussian_opacities(logit_opacities, one_camera), tiles
            )
        # The class method takes precomputed tiles for repeated crops and renders the same pixels with them.
        model = self._model(params)
        pg = model.project_gaussians_for_images(self.w2c, self.K, self.W, self.H, 0.01, 1e10)
        pg_tiles = pg.tile_intersection(16)
        crop = dict(crop_width=40, crop_height=30, crop_origin_w=5, crop_origin_h=7)
        without, _ = model.render_from_projected_gaussians(pg, **crop)
        with_tiles, _ = model.render_from_projected_gaussians(pg, tiles=pg_tiles, **crop)
        torch.testing.assert_close(with_tiles, without)
        # Only negative sizes mean "full image"; zero is a malformed crop.
        with self.assertRaisesRegex(ValueError, "positive"):
            model.render_from_projected_gaussians(pg, crop_width=0, crop_height=30)
        # The tile size comes from the tiles when they are given; an explicit size that disagrees is refused.
        with_eight, _ = model.render_from_projected_gaussians(pg, tiles=pg.tile_intersection(8), **crop)
        torch.testing.assert_close(with_eight, without, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "tile_size"):
            model.render_from_projected_gaussians(pg, tiles=pg_tiles, tile_size=8, **crop)


if __name__ == "__main__":
    unittest.main()
