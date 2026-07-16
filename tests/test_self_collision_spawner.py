from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


SPAWNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "openarm_testusd_task"
    / "openarm_testusd_task"
    / "tasks"
    / "testusd_reach"
    / "original_usd_spawner.py"
)

_ARTICULATION_ROOT_API = object()
_RIGID_BODY_API = object()


class _Attribute:
    def __init__(self, value: bool, *, accepts_write: bool = True, path: str = "/Robot/root_joint.selfCollision"):
        self.value = value
        self.accepts_write = accepts_write
        self.path = path

    def Set(self, value: bool) -> bool:
        if self.accepts_write:
            self.value = value
        return self.accepts_write

    def Get(self) -> bool:
        return self.value

    def GetPath(self) -> str:
        return self.path


class _ArticulationApi:
    def __init__(self, attribute: _Attribute):
        self.attribute = attribute

    def CreateEnabledSelfCollisionsAttr(self) -> _Attribute:
        return self.attribute

    def GetEnabledSelfCollisionsAttr(self) -> _Attribute:
        return self.attribute


class _Relationship:
    def __init__(self, targets=(), path: str = "/Robot.physics:filteredPairs"):
        self.targets = list(targets)
        self.path = path

    def AddTarget(self, target) -> bool:
        if target not in self.targets:
            self.targets.append(target)
        return True

    def SetTargets(self, targets) -> bool:
        self.targets = list(targets)
        return True

    def GetTargets(self):
        return list(self.targets)

    def GetPath(self) -> str:
        return self.path

    @property
    def targetPathList(self):
        return SimpleNamespace(GetAppliedItems=lambda: list(self.targets))


class _FilteredPairsApi:
    def __init__(self, relationship: _Relationship):
        self.relationship = relationship

    def CreateFilteredPairsRel(self) -> _Relationship:
        return self.relationship

    def GetFilteredPairsRel(self) -> _Relationship:
        return self.relationship


class _Prim:
    def __init__(
        self,
        *,
        valid: bool = True,
        instance_proxy: bool = False,
        articulation_root: bool = True,
        rigid_body: bool = True,
        api: _ArticulationApi | None = None,
        filtered_pairs_api: _FilteredPairsApi | None = None,
        path: str = "/Robot",
        descendants: list[_Prim] | None = None,
        api_schemas: set[str] | None = None,
    ):
        self.valid = valid
        self.instance_proxy = instance_proxy
        self.articulation_root = articulation_root
        self.rigid_body = rigid_body
        self.physx_api = api
        self.filtered_pairs_api = filtered_pairs_api
        self.path = path
        self.descendants = descendants if descendants is not None else [self]
        self.api_schemas = set(api_schemas or ())

    def IsValid(self) -> bool:
        return self.valid

    def IsInstanceProxy(self) -> bool:
        return self.instance_proxy

    def HasAPI(self, api_type) -> bool:
        if api_type is _RIGID_BODY_API:
            return self.rigid_body
        return self.articulation_root

    def GetPath(self) -> str:
        return self.path

    def HasInfo(self, name: str) -> bool:
        return name == "apiSchemas" and bool(self.api_schemas)

    def GetInfo(self, name: str):
        if name != "apiSchemas":
            return None
        return SimpleNamespace(GetAppliedItems=lambda: list(self.api_schemas))


class _Stage:
    def __init__(self, prim: _Prim, *, root_authored: bool = True):
        self.prim = prim
        self.root_authored = root_authored

    def GetPrimAtPath(self, path: str) -> _Prim:
        return self.prim

    def GetRootLayer(self):
        return self

    def GetAttributeAtPath(self, path: str):
        if not self.root_authored or self.prim.physx_api is None:
            return None
        return SimpleNamespace(default=self.prim.physx_api.attribute.value)


class _ContractStage:
    def __init__(self, prims: dict[str, _Prim]):
        self.prims = prims

    def GetPrimAtPath(self, path: str) -> _Prim:
        return self.prims.get(path, _Prim(valid=False))


class _FilterAuthorStage(_ContractStage):
    def GetRootLayer(self):
        return self

    def GetRelationshipAtPath(self, path: str):
        for prim in self.prims.values():
            api = prim.filtered_pairs_api
            if api is not None and api.relationship.GetPath() == path:
                return api.relationship
        return None


class _PhysxArticulationApiFactory:
    def __new__(cls, prim: _Prim):
        return prim.physx_api

    @staticmethod
    def Apply(prim: _Prim) -> _ArticulationApi:
        prim.physx_api = _ArticulationApi(_Attribute(False))
        return prim.physx_api


class _FilteredPairsApiFactory:
    def __new__(cls, prim: _Prim):
        return prim.filtered_pairs_api

    @staticmethod
    def Apply(prim: _Prim) -> _FilteredPairsApi:
        prim.api_schemas.add("PhysicsFilteredPairsAPI")
        if prim.filtered_pairs_api is None:
            prim.filtered_pairs_api = _FilteredPairsApi(
                _Relationship(path=f"{prim.GetPath()}.physics:filteredPairs")
            )
        return prim.filtered_pairs_api


def _load_helper():
    tree = ast.parse(SPAWNER_PATH.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_enable_articulation_self_collisions"
    )
    namespace = {
        "Usd": SimpleNamespace(
            Stage=object,
            EditContext=lambda stage, edit_target: nullcontext(),
            EditTarget=lambda layer: layer,
        ),
        "UsdPhysics": SimpleNamespace(ArticulationRootAPI=_ARTICULATION_ROOT_API),
        "PhysxSchema": SimpleNamespace(
            PhysxArticulationAPI=_PhysxArticulationApiFactory
        ),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(SPAWNER_PATH), "exec"),
        namespace,
    )
    return namespace[helper.name]


def _load_contract():
    tree = ast.parse(SPAWNER_PATH.read_text())
    contract = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "assert_openarm_self_collision_contract"
    )
    namespace = {
        "Usd": SimpleNamespace(PrimRange=lambda prim: prim.descendants),
        "UsdPhysics": SimpleNamespace(
            ArticulationRootAPI=_ARTICULATION_ROOT_API,
            RigidBodyAPI=_RIGID_BODY_API,
            FilteredPairsAPI=_FilteredPairsApiFactory,
        ),
        "Sdf": SimpleNamespace(Path=lambda value: value),
        "PhysxSchema": SimpleNamespace(
            PhysxArticulationAPI=_PhysxArticulationApiFactory
        ),
        "OPENARM_ROBOT_REL_PATH": "openarm_bimanual",
        "OPENARM_ARTICULATION_ROOT_PRIM_PATH": "/root_joint",
        "OPENARM_ARTICULATION_REL_PATH": "openarm_bimanual/root_joint",
        "OPENARM_SELF_COLLISION_PROFILE_MAX_ENVS": 7040,
        "OPENARM_SELF_COLLISION_FILTER_PAIRS": (
            ("openarm_left_link5", "openarm_left_link7"),
            ("openarm_right_link5", "openarm_right_link7"),
        ),
    }
    exec(
        compile(ast.Module(body=[contract], type_ignores=[]), str(SPAWNER_PATH), "exec"),
        namespace,
    )
    return namespace[contract.name]


def _load_filter_helper():
    tree = ast.parse(SPAWNER_PATH.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_author_openarm_self_collision_filters"
    )
    namespace = {
        "Usd": SimpleNamespace(
            Stage=object,
            EditContext=lambda stage, edit_target: nullcontext(),
            EditTarget=lambda layer: layer,
        ),
        "UsdPhysics": SimpleNamespace(
            RigidBodyAPI=_RIGID_BODY_API,
            FilteredPairsAPI=_FilteredPairsApiFactory,
        ),
        "Sdf": SimpleNamespace(Path=lambda value: value),
        "OPENARM_SELF_COLLISION_FILTER_PAIRS": (
            ("openarm_left_link5", "openarm_left_link7"),
            ("openarm_right_link5", "openarm_right_link7"),
        ),
    }
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), str(SPAWNER_PATH), "exec"),
        namespace,
    )
    return namespace[helper.name]


def _contract_env(
    attributes: list[_Attribute],
    *,
    robot_prim_path: str = "/World/envs/env_.*/Scene/openarm_bimanual",
    root_prim_path: str = "/root_joint",
    spawn=None,
    missing_filter: tuple[int, str] | None = None,
):
    env_paths = [f"/World/envs/env_{index}" for index in range(len(attributes))]
    prims = {
        f"{env_path}/Scene/openarm_bimanual/root_joint": _Prim(
            api=_ArticulationApi(attribute),
            path=f"{env_path}/Scene/openarm_bimanual/root_joint",
        )
        for env_path, attribute in zip(env_paths, attributes, strict=True)
    }
    for env_index, env_path in enumerate(env_paths):
        robot_path = f"{env_path}/Scene/openarm_bimanual"
        robot_links: list[_Prim] = []
        for source_link, target_link in (
            ("openarm_left_link5", "openarm_left_link7"),
            ("openarm_right_link5", "openarm_right_link7"),
        ):
            targets = []
            if missing_filter != (env_index, source_link):
                targets.append(f"{robot_path}/{target_link}")
            source_path = f"{robot_path}/{source_link}"
            target_path = f"{robot_path}/{target_link}"
            source_prim = _Prim(
                filtered_pairs_api=_FilteredPairsApi(
                    _Relationship(
                        targets,
                        path=f"{source_path}.physics:filteredPairs",
                    )
                ),
                path=source_path,
                api_schemas={"PhysicsFilteredPairsAPI"},
            )
            target_prim = _Prim(path=target_path)
            prims[source_path] = source_prim
            prims[target_path] = target_prim
            robot_links.extend((source_prim, target_prim))
        robot_prim = _Prim(path=robot_path, descendants=[])
        robot_prim.descendants = [robot_prim, *robot_links]
        prims[robot_path] = robot_prim
    robot_cfg = SimpleNamespace(
        prim_path=robot_prim_path,
        articulation_root_prim_path=root_prim_path,
        spawn=spawn,
    )
    scene = SimpleNamespace(
        cfg=SimpleNamespace(robot=robot_cfg),
        env_regex_ns="/World/envs/env_.*",
        env_prim_paths=env_paths,
        stage=_ContractStage(prims),
    )
    return SimpleNamespace(scene=scene)


def test_self_collision_helper_overrides_false_and_verifies_true():
    attribute = _Attribute(False)
    prim = _Prim(api=_ArticulationApi(attribute))

    _load_helper()(_Stage(prim), "/Robot/root_joint")

    assert attribute.value is True


def test_self_collision_helper_applies_missing_physx_api():
    prim = _Prim(api=None)

    _load_helper()(_Stage(prim), "/Robot/root_joint")

    assert prim.physx_api is not None
    assert prim.physx_api.attribute.value is True


def test_self_collision_helper_requires_root_layer_opinion():
    prim = _Prim(api=_ArticulationApi(_Attribute(False)))

    with pytest.raises(RuntimeError, match="Failed to enable"):
        _load_helper()(_Stage(prim, root_authored=False), "/Robot/root_joint")


@pytest.mark.parametrize(
    ("prim", "error_type", "message"),
    [
        (_Prim(valid=False), ValueError, "does not exist"),
        (_Prim(instance_proxy=True), RuntimeError, "instance proxy"),
        (_Prim(articulation_root=False), RuntimeError, "not an articulation root"),
        (
            _Prim(api=_ArticulationApi(_Attribute(False, accepts_write=False))),
            RuntimeError,
            "Failed to enable",
        ),
    ],
)
def test_self_collision_helper_fails_closed(prim, error_type, message):
    with pytest.raises(error_type, match=message):
        _load_helper()(_Stage(prim), "/Robot/root_joint")


def test_filter_helper_authors_only_the_validated_bilateral_link_pairs():
    prims = {
        f"/Robot/openarm_{side}_link{link}": _Prim(
            path=f"/Robot/openarm_{side}_link{link}"
        )
        for side in ("left", "right")
        for link in (5, 7)
    }
    prims["/Robot/openarm_left_link5"].filtered_pairs_api = _FilteredPairsApi(
        _Relationship(
            ["/Robot/unexpected_link"],
            path="/Robot/openarm_left_link5.physics:filteredPairs",
        )
    )
    stage = _FilterAuthorStage(prims)

    _load_filter_helper()(stage, "/Robot")

    assert prims["/Robot/openarm_left_link5"].filtered_pairs_api.relationship.targets == [
        "/Robot/openarm_left_link7"
    ]
    assert prims["/Robot/openarm_right_link5"].filtered_pairs_api.relationship.targets == [
        "/Robot/openarm_right_link7"
    ]
    assert prims["/Robot/openarm_left_link7"].filtered_pairs_api is None
    assert prims["/Robot/openarm_right_link7"].filtered_pairs_api is None


def test_filter_helper_fails_closed_when_a_profiled_link_is_missing():
    prims = {
        f"/Robot/openarm_{side}_link{link}": _Prim(
            path=f"/Robot/openarm_{side}_link{link}"
        )
        for side in ("left", "right")
        for link in (5, 7)
    }
    prims["/Robot/openarm_right_link7"] = _Prim(
        valid=False,
        path="/Robot/openarm_right_link7",
    )

    with pytest.raises(RuntimeError, match="not a rigid body"):
        _load_filter_helper()(_FilterAuthorStage(prims), "/Robot")


def test_startup_contract_accepts_only_when_every_clone_is_enabled():
    env = _contract_env([_Attribute(True), _Attribute(True), _Attribute(True)])

    _load_contract()(env, None)


def test_startup_contract_rejects_one_disabled_clone():
    env = _contract_env([_Attribute(True), _Attribute(False), _Attribute(True)])

    with pytest.raises(RuntimeError, match="1 OpenArm articulation"):
        _load_contract()(env, None)


def test_startup_contract_rejects_one_missing_link_filter():
    env = _contract_env(
        [_Attribute(True), _Attribute(True)],
        missing_filter=(1, "openarm_right_link5"),
    )

    with pytest.raises(RuntimeError, match="1 OpenArm link pair"):
        _load_contract()(env, None)


def test_startup_contract_rejects_a_dangling_profiled_target():
    env = _contract_env([_Attribute(True)])
    target_path = (
        "/World/envs/env_0/Scene/openarm_bimanual/openarm_right_link7"
    )
    env.scene.stage.prims[target_path].valid = False

    with pytest.raises(RuntimeError, match="filters are invalid"):
        _load_contract()(env, None)


def test_startup_contract_rejects_an_extra_target_on_a_profiled_source():
    env = _contract_env([_Attribute(True)])
    source_path = (
        "/World/envs/env_0/Scene/openarm_bimanual/openarm_left_link5"
    )
    env.scene.stage.prims[source_path].filtered_pairs_api.relationship.targets.append(
        "/World/envs/env_0/Scene/openarm_bimanual/openarm_left_link3"
    )

    with pytest.raises(RuntimeError, match="filters are invalid"):
        _load_contract()(env, None)


def test_startup_contract_rejects_a_filter_on_an_unprofiled_link():
    env = _contract_env([_Attribute(True)])
    robot_path = "/World/envs/env_0/Scene/openarm_bimanual"
    extra_path = f"{robot_path}/openarm_left_link3"
    extra_prim = _Prim(
        path=extra_path,
        filtered_pairs_api=_FilteredPairsApi(
            _Relationship(
                [f"{robot_path}/openarm_left_link6"],
                path=f"{extra_path}.physics:filteredPairs",
            )
        ),
        api_schemas={"PhysicsFilteredPairsAPI"},
    )
    env.scene.stage.prims[extra_path] = extra_prim
    env.scene.stage.prims[robot_path].descendants.append(extra_prim)

    with pytest.raises(RuntimeError, match="filters are invalid"):
        _load_contract()(env, None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"robot_prim_path": "/World/envs/env_.*/Scene/other_robot"},
        {"root_prim_path": "/other_root"},
        {"spawn": object()},
    ],
)
def test_startup_contract_rejects_robot_binding_drift(overrides):
    env = _contract_env([_Attribute(True)], **overrides)

    with pytest.raises(RuntimeError, match="robot binding"):
        _load_contract()(env, None)


def test_startup_contract_rejects_unprofiled_environment_count():
    env = _contract_env([_Attribute(True)] * 7041)

    with pytest.raises(RuntimeError, match="has not been validated above 7040"):
        _load_contract()(env, None)
