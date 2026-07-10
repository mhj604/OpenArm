from __future__ import annotations

from dataclasses import MISSING

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

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
    translation_overrides: dict[str, tuple[float, float, float]] | None = None
    disable_gravity_paths: tuple[str, ...] = ()


def _reference_prim(stage: Usd.Stage, target_path: str, usd_path: str, source_path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(target_path)
    if not prim.IsValid():
        prim = stage.OverridePrim(target_path)
    prim.GetReferences().AddReference(Sdf.Reference(usd_path, source_path))
    return prim


def _disable_rigid_body_gravity(stage: Usd.Stage, root_path: str) -> None:
    """Author gravity overrides on every rigid body below a referenced prim."""

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"Gravity override root prim does not exist: {root_path}")

    rigid_body_count = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not rigid_body_api:
            rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_body_api.CreateDisableGravityAttr().Set(True)
        rigid_body_count += 1

    if rigid_body_count == 0:
        raise RuntimeError(f"No rigid bodies found below gravity override root: {root_path}")


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

    for relative_path, override_translation in (cfg.translation_overrides or {}).items():
        target_path = f"{prim_path.rstrip('/')}/{relative_path.lstrip('/')}"
        target_prim = stage.OverridePrim(target_path)
        translate_attr = target_prim.GetAttribute("xformOp:translate")
        if not translate_attr.IsValid():
            translate_attr = UsdGeom.Xformable(target_prim).AddTranslateOp().GetAttr()
        translate_attr.Set(override_translation)

    for relative_path in cfg.disable_gravity_paths:
        target_path = f"{prim_path.rstrip('/')}/{relative_path.lstrip('/')}"
        _disable_rigid_body_gravity(stage, target_path)

    # These are global stage-level prims in the original file. They should not be nested under env_0.
    _reference_prim(stage, cfg.environment_path, cfg.usd_path, "/Environment")
    _reference_prim(stage, cfg.render_path, cfg.usd_path, "/Render")

    # Preserve the original stage's basic metadata.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    return scene_prim
