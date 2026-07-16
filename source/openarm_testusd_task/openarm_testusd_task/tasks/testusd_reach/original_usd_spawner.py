from __future__ import annotations

from dataclasses import MISSING

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage, standardize_xform_ops
from isaaclab.utils import configclass


@configclass
class OriginalUsdSceneCfg(SpawnerCfg):
    """Reference the original OpenArm scene while preserving non-default root prims."""

    func = MISSING
    usd_path: str = MISSING
    environment_path: str = "/Environment"
    render_path: str = "/Render"
    translation_overrides: dict[str, tuple[float, float, float]] | None = None
    disable_gravity_paths: tuple[str, ...] = ()


OPENARM_ROBOT_REL_PATH = "openarm_bimanual"
OPENARM_ARTICULATION_ROOT_PRIM_PATH = "/root_joint"
OPENARM_ARTICULATION_REL_PATH = OPENARM_ROBOT_REL_PATH + OPENARM_ARTICULATION_ROOT_PRIM_PATH
OPENARM_SELF_COLLISION_PROFILE_MAX_ENVS = 7040
OPENARM_SELF_COLLISION_FILTER_PAIRS = (
    ("openarm_left_link5", "openarm_left_link7"),
    ("openarm_right_link5", "openarm_right_link7"),
)


def _reference_prim(stage: Usd.Stage, target_path: str, usd_path: str, source_path: str) -> Usd.Prim:
    source_layer = Sdf.Layer.FindOrOpen(usd_path)
    if source_layer is None:
        raise FileNotFoundError(f"Unable to open referenced USD: {usd_path}")
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        prim = stage.GetPrimAtPath(target_path)
        if not prim.IsValid():
            prim = stage.OverridePrim(target_path)
        references = prim.GetReferences()
        references.ClearReferences()
        reference_succeeded = references.AddReference(Sdf.Reference(usd_path, source_path))
    if not reference_succeeded:
        raise RuntimeError(f"Failed to reference {source_path} from USD: {usd_path}")
    expected_source_path = Sdf.Path(source_path)
    if not any(
        spec.layer.identifier == source_layer.identifier and spec.path == expected_source_path
        for spec in prim.GetPrimStack()
    ):
        raise RuntimeError(f"Referenced source did not contribute to {target_path}: {usd_path}{source_path}")
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


def _enable_articulation_self_collisions(stage: Usd.Stage, articulation_path: str) -> None:
    """Author and verify self-collisions on an exact articulation root prim."""

    articulation_prim = stage.GetPrimAtPath(articulation_path)
    if not articulation_prim.IsValid():
        raise ValueError(f"Self-collision articulation prim does not exist: {articulation_path}")
    if articulation_prim.IsInstanceProxy():
        raise RuntimeError(f"Cannot author self-collisions on an instance proxy: {articulation_path}")
    if not articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(f"Prim is not an articulation root: {articulation_path}")

    # @clone copies the source prim spec from the root layer by default. Author
    # there explicitly so a caller's temporary edit target cannot drop this
    # override from cloned environments.
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
        if not articulation_api:
            articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
        enabled_attr = articulation_api.CreateEnabledSelfCollisionsAttr()
        write_succeeded = enabled_attr.Set(True)
    root_attr_spec = stage.GetRootLayer().GetAttributeAtPath(enabled_attr.GetPath())
    if (
        not write_succeeded
        or root_attr_spec is None
        or root_attr_spec.default is not True
        or enabled_attr.Get() is not True
    ):
        raise RuntimeError(f"Failed to enable articulation self-collisions: {articulation_path}")


def _author_openarm_self_collision_filters(stage: Usd.Stage, robot_path: str) -> None:
    """Filter only the close kinematic link pairs validated for the OpenArm asset."""

    for source_link, target_link in OPENARM_SELF_COLLISION_FILTER_PAIRS:
        source_path = f"{robot_path.rstrip('/')}/{source_link}"
        target_path = f"{robot_path.rstrip('/')}/{target_link}"
        source_prim = stage.GetPrimAtPath(source_path)
        target_prim = stage.GetPrimAtPath(target_path)
        for link_path, link_prim in ((source_path, source_prim), (target_path, target_prim)):
            if not link_prim.IsValid() or not link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Self-collision filter link is not a rigid body: {link_path}")

        with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
            filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
            target_sdf_path = Sdf.Path(target_path)
            write_succeeded = filtered_pairs_rel.SetTargets([target_sdf_path])

        root_layer = stage.GetRootLayer()
        root_prim_spec = root_layer.GetPrimAtPath(source_prim.GetPath())
        root_relationship = root_layer.GetRelationshipAtPath(filtered_pairs_rel.GetPath())
        root_api_schemas = (
            root_prim_spec.GetInfo("apiSchemas")
            if root_prim_spec is not None and root_prim_spec.HasInfo("apiSchemas")
            else None
        )
        root_applied_schemas = (
            set(root_api_schemas.GetAppliedItems())
            if root_api_schemas is not None
            else set()
        )
        root_targets = (
            set(root_relationship.targetPathList.GetAppliedItems())
            if root_relationship is not None
            else set()
        )
        if (
            not write_succeeded
            or "PhysicsFilteredPairsAPI" not in root_applied_schemas
            or root_relationship is None
            or root_targets != {target_sdf_path}
            or set(filtered_pairs_rel.GetTargets()) != {target_sdf_path}
        ):
            raise RuntimeError(
                f"Failed to author OpenArm self-collision filter: {source_path} -> {target_path}"
            )


def apply_openarm_self_collision_physx_profile(physx_cfg) -> None:
    """Reserve the shared contact-rich PhysX profile used by OpenArm tasks."""

    physx_cfg.gpu_max_rigid_contact_count = 2**23
    physx_cfg.gpu_max_rigid_patch_count = 2**23
    physx_cfg.gpu_found_lost_pairs_capacity = 2**22
    physx_cfg.gpu_found_lost_aggregate_pairs_capacity = 2**25
    physx_cfg.gpu_total_aggregate_pairs_capacity = 2**21
    physx_cfg.gpu_collision_stack_size = 2**28
    physx_cfg.gpu_heap_capacity = 2**27
    physx_cfg.gpu_temp_buffer_capacity = 2**25


def assert_openarm_self_collision_contract(env, env_ids) -> None:
    """Abort startup unless every cloned OpenArm articulation has self-collisions enabled."""

    if env_ids is not None:
        raise RuntimeError("OpenArm self-collision startup verification must cover every environment.")

    robot_cfg = env.scene.cfg.robot
    expected_robot_path = f"{env.scene.env_regex_ns}/Scene/{OPENARM_ROBOT_REL_PATH}"
    if (
        robot_cfg.prim_path != expected_robot_path
        or robot_cfg.articulation_root_prim_path != OPENARM_ARTICULATION_ROOT_PRIM_PATH
        or robot_cfg.spawn is not None
    ):
        raise RuntimeError(
            "OpenArm robot binding no longer matches the mandatory self-collision articulation: "
            f"prim_path={robot_cfg.prim_path}, root={robot_cfg.articulation_root_prim_path}, "
            f"spawn={robot_cfg.spawn}"
        )

    env_prim_paths = env.scene.env_prim_paths
    if len(env_prim_paths) > OPENARM_SELF_COLLISION_PROFILE_MAX_ENVS:
        raise RuntimeError(
            "OpenArm self-collision PhysX profile has not been validated above "
            f"{OPENARM_SELF_COLLISION_PROFILE_MAX_ENVS} environments."
        )

    stage = env.scene.stage
    invalid_paths: list[str] = []
    invalid_filters: list[str] = []
    for env_prim_path in env_prim_paths:
        articulation_path = f"{env_prim_path}/Scene/{OPENARM_ARTICULATION_REL_PATH}"
        articulation_prim = stage.GetPrimAtPath(articulation_path)
        if not articulation_prim.IsValid() or not articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            invalid_paths.append(articulation_path)
            continue
        articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
        enabled_attr = articulation_api.GetEnabledSelfCollisionsAttr() if articulation_api else None
        if not enabled_attr or enabled_attr.Get() is not True:
            invalid_paths.append(articulation_path)

        robot_path = f"{env_prim_path}/Scene/{OPENARM_ROBOT_REL_PATH}"
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            invalid_filters.append(f"missing robot prim: {robot_path}")
            continue
        expected_filters = {
            Sdf.Path(f"{robot_path}/{source_link}"): {
                Sdf.Path(f"{robot_path}/{target_link}")
            }
            for source_link, target_link in OPENARM_SELF_COLLISION_FILTER_PAIRS
        }
        observed_filters: dict[Sdf.Path, set[Sdf.Path]] = {}
        for prim in Usd.PrimRange(robot_prim):
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI(prim)
            if not filtered_pairs_api:
                continue
            filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
            targets = set(filtered_pairs_rel.GetTargets()) if filtered_pairs_rel else set()
            if targets:
                observed_filters[prim.GetPath()] = targets

        for source_link, target_link in OPENARM_SELF_COLLISION_FILTER_PAIRS:
            source_path = f"{robot_path}/{source_link}"
            target_path = f"{robot_path}/{target_link}"
            source_prim = stage.GetPrimAtPath(source_path)
            target_prim = stage.GetPrimAtPath(target_path)
            if (
                not source_prim.IsValid()
                or not target_prim.IsValid()
                or not source_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not target_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                invalid_filters.append(f"{source_path} -> {target_path}")
        if observed_filters != expected_filters:
            invalid_filters.append(
                f"unexpected filter set under {robot_path}: {observed_filters}"
            )

    if invalid_paths:
        preview = ", ".join(invalid_paths[:8])
        raise RuntimeError(
            f"Self-collisions are not enabled on {len(invalid_paths)} OpenArm articulation(s): {preview}"
        )
    if invalid_filters:
        preview = ", ".join(invalid_filters[:8])
        raise RuntimeError(
            f"Self-collision filters are invalid on {len(invalid_filters)} OpenArm link pair(s): {preview}"
        )


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
    articulation_path = f"{prim_path.rstrip('/')}/{OPENARM_ARTICULATION_REL_PATH}"
    _enable_articulation_self_collisions(stage, articulation_path)
    robot_path = f"{prim_path.rstrip('/')}/{OPENARM_ROBOT_REL_PATH}"
    _author_openarm_self_collision_filters(stage, robot_path)
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
