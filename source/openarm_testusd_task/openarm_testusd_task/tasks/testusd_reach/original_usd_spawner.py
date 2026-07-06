from __future__ import annotations

from dataclasses import MISSING

from pxr import Sdf, Usd, UsdGeom

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage, standardize_xform_ops
from isaaclab.utils import configclass


@configclass
class OriginalUsdSceneCfg(SpawnerCfg):
    """Reference the original scene USD while preserving non-default root prims."""

    func = MISSING
    usd_path: str = MISSING
    environment_path: str = "/Environment"
    render_path: str = "/Render"


def _reference_prim(stage: Usd.Stage, target_path: str, usd_path: str, source_path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        prim = stage.OverridePrim(target_path)
    prim.GetReferences().AddReference(Sdf.Reference(usd_path, source_path))
    return prim


@clone
def spawn_original_usd_scene(
    prim_path: str,
    cfg: OriginalUsdSceneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn original `/World` under the env and reference `/Environment` and `/Render` globally."""

    stage = get_current_stage()
    source_layer = Sdf.Layer.FindOrOpen(cfg.usd_path)
    if source_layer is None:
        raise FileNotFoundError(f"Unable to open original USD: {cfg.usd_path}")

    default_prim = source_layer.defaultPrim or "World"
    scene_prim = _reference_prim(stage, prim_path, cfg.usd_path, f"/{default_prim}")
    standardize_xform_ops(
        scene_prim,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    # These are global stage-level prims in the original file. They should not be nested under env_0.
    _reference_prim(stage, cfg.environment_path, cfg.usd_path, "/Environment")
    _reference_prim(stage, cfg.render_path, cfg.usd_path, "/Render")

    # Preserve the original stage's basic metadata.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    return scene_prim
