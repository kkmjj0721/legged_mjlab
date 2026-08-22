import math
import os

import mujoco

from legged_mjlab.utils.paths import PROJECT_ROOT, resource_path
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

from .him_go2_config import HimGo2RounghCfg

class Go2Asset:
    """Go2 的 MJCF、执行器、碰撞和传感器聚合器。

    EntityCfg 只描述实体/MJCF/actuator/collision。
    SensorCfg 由 build_scene_sensors() 交给 SceneCfg.sensors。
    """
    class Names:
        entity = "robot"
        base_body = "base_link"
        imu_site = "imu"

        foot_sites = ("FL", "FR", "RL", "RR")
        foot_geoms = tuple(f"{name}_foot_collision" for name in foot_sites)
        joint_order = (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        )

        joint_groups = {
            "hip": (r".*_hip_joint",),
            "thigh": (r".*_thigh_joint",),
            "calf": (r".*_calf_joint",),
        }

    FOOT_GEOM_REGEX = r"^(FL|FR|RL|RR)_foot_collision$"
    COLLISION_REGEX = r".*_collision$"

    class SensorMetadata:
        def __init__(
            self,
            name,
            mount,
            frame,
            unit,
            shape,
            source,
            required=True,
            finite_required=True,
            max_age_s=None,
            channels=(),
            field_shapes=(),
        ):
            self.name = name
            self.mount = mount
            self.frame = frame
            self.unit = unit
            self.shape = shape
            self.source = source
            self.required = required
            self.finite_required = finite_required
            self.max_age_s = max_age_s
            self.channels = channels
            self.field_shapes = field_shapes


    def __init__(self, cfg: HimGo2RounghCfg, *, entity_name: str | None = None, include_default_scene_sensors: bool = False,):
        self.cfg = cfg
        self.entity_name = entity_name or self.Names.entity
        self.xml_path = self._resolve_xml_path(cfg.asset.xml)

        self.actuator_cfgs = []
        self.collision_cfgs = []
        self.scene_sensor_cfgs = []
        self.sensor_metadata = {}

        self._configure_default_actuators()
        self._configure_default_collisions()
        self._register_mjcf_sensor_metadata()
        if include_default_scene_sensors:
            self._configure_default_scene_sensors()

    def _resolve_xml_path(self, raw_xml):
        raw = str(raw_xml).replace("{LEGGED_MJLAB_ROOT_DIR}", str(PROJECT_ROOT))
        raw = os.path.expanduser(raw)
        if not os.path.isabs(raw):
            raw = os.path.join(str(PROJECT_ROOT), raw)
        path = os.path.abspath(raw)
        if not os.path.isfile(path):
            try:
                res_path = resource_path(raw_xml)
                if os.path.isfile(str(res_path)):
                    return os.path.abspath(str(res_path))
            except Exception:
                pass
            raise FileNotFoundError(f"Go2 MJCF does not exist: {path}")
        return path

    def _collect_xml_assets(self, spec):
        xml_dir = os.path.dirname(self.xml_path)
        mesh_root = os.path.abspath(os.path.join(xml_dir, spec.meshdir))
        if not os.path.isdir(mesh_root):
            raise FileNotFoundError(
                f"meshdir={spec.meshdir!r} resolves to missing directory: {mesh_root}"
            )
        assets = {}
        for root, _, files in os.walk(mesh_root):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, mesh_root).replace("\\", "/")
                with open(full_path, "rb") as f:
                    assets[rel_path] = f.read()
        return assets

    def build_spec(self):
        spec = mujoco.MjSpec.from_file(str(self.xml_path))
        self.validate_canonical_spec(spec)
        spec.assets = self._collect_xml_assets(spec)
        return spec

    def validate_canonical_spec(self, spec):
        if os.path.basename(self.xml_path) != "go2.xml":
            raise ValueError(
                f"unsupported Go2 asset source: {self.xml_path}; "
                "expected canonical go2.xml"
            )
        if len(spec.actuators) != 0:
            raise ValueError(
                "canonical go2.xml must not contain MJCF actuators; "
                "actuators must be configured declaratively via EntityCfg"
            )

    def _configure_default_actuators(self):
        """配置驱动器参数。
        """
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping
        armature = getattr(self.cfg.asset, "armature", None)
        effort_limit = getattr(self.cfg.asset, "effort_limit", None)

        for group_name, target_names_expr in self.Names.joint_groups.items():
            self.add_actuator(
                BuiltinPositionActuatorCfg(
                    target_names_expr=target_names_expr,
                    stiffness=stiffness[group_name],
                    damping=damping[group_name],
                    armature=armature,
                    effort_limit=effort_limit,
                )
            )

    def _configure_default_collisions(self):
        """配置碰撞体过滤与足端摩擦矩阵。
        """
        self.add_collision(
            CollisionCfg(
                geom_names_expr=(self.COLLISION_REGEX,),
                contype=1,
                conaffinity=1,
                condim={
                    self.FOOT_GEOM_REGEX: 3,
                    self.COLLISION_REGEX: 1,
                },
                priority=0,
                friction={self.FOOT_GEOM_REGEX: (0.6,)},
            )
        )

    def add_actuator(self, actuator_cfg):
        self.actuator_cfgs.append(actuator_cfg)

    def add_collision(self, collision_cfg):
        self.collision_cfgs.append(collision_cfg)

    def _register_mjcf_sensor_metadata(self):
        """注册 MJCF 中内置的传感器
        """
        self._register_metadata(
            self.SensorMetadata(
                name="imu_ang_vel",
                mount=self.Names.imu_site,
                frame="imu",
                unit="rad/s",
                shape=(3,),
                source="MJCF sensor <gyro>",
                channels=("value",),
            )
        )
        self._register_metadata(
            self.SensorMetadata(
                name="imu_lin_vel",
                mount=self.Names.imu_site,
                frame="imu",
                unit="m/s",
                shape=(3,),
                source="MJCF sensor <velocimeter>",
                channels=("value",),
            )
        )
        self._register_metadata(
            self.SensorMetadata(
                name="imu_lin_acc",
                mount=self.Names.imu_site,
                frame="imu",
                unit="m/s^2",
                shape=(3,),
                source="MJCF sensor <accelerometer>",
                channels=("value",),
            )
        )
        self._register_metadata(
            self.SensorMetadata(
                name="root_angmom",
                mount=self.Names.base_body,
                frame="world",
                unit="kg*m^2/s",
                shape=(3,),
                source="MJCF sensor <subtreeangmom>",
                channels=("value",),
            )
        )

    def _configure_default_scene_sensors(self):
        """配置默认的 Scene 级足端触地传感器。"""
        self.add_foot_contact_sensor()

    def _register_metadata(self, metadata):
        if not metadata.name:
            raise ValueError("sensor metadata name must not be empty")
        if metadata.name in self.sensor_metadata:
            raise ValueError(f"duplicate Go2 sensor metadata: {metadata.name}")
        self.sensor_metadata[metadata.name] = metadata

    def add_sensor(self, sensor_cfg, *, metadata=None, replace=False):
        """添加 Scene 级传感器配置并注册元数据。"""
        existing = {
            sensor.name: index
            for index, sensor in enumerate(self.scene_sensor_cfgs)
        }
        if not sensor_cfg.name:
            raise ValueError("Scene sensor name must not be empty")
        if sensor_cfg.name in existing and not replace:
            raise ValueError(f"duplicate Scene sensor: {sensor_cfg.name}")

        if sensor_cfg.name in existing:
            self.scene_sensor_cfgs[existing[sensor_cfg.name]] = sensor_cfg
        else:
            self.scene_sensor_cfgs.append(sensor_cfg)

        if metadata is not None:
            if metadata.name != sensor_cfg.name:
                raise ValueError("sensor config and metadata names must match")
            if replace:
                self.sensor_metadata.pop(metadata.name, None)
            self._register_metadata(metadata)

    def add_foot_contact_sensor(self):
        """添加四足足端地面接触力/状态传感器。"""
        self.add_sensor(
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=self.FOOT_GEOM_REGEX,
                    entity=self.entity_name,
                ),
                secondary=ContactMatch(
                    mode="body",
                    pattern="terrain",
                ),
                fields=("found", "force"),
                reduce="netforce",
                track_air_time=True,
            ),
            metadata=self.SensorMetadata(
                name="feet_ground_contact",
                mount="FL/FR/RL/RR foot geoms",
                frame="world",
                unit="found:bool, force:N",
                shape=(16,),
                source="Scene ContactSensorCfg",
                channels=("found", "force"),
                field_shapes=(("found", (4,)), ("force", (4, 3))),
                max_age_s=0.02,
            ),
        )

    def build_scene_sensors(self):
        """返回构建 Scene 所需的全部 SensorCfg 元组。"""
        return tuple(self.scene_sensor_cfgs)

    def build_entity_cfg(self):
        """实体构建导出，生成标准 mjlab EntityCfg 配置对象。"""
        init_state = EntityCfg.InitialStateCfg(
            pos=tuple(self.cfg.init_state.pos),
            joint_pos=dict(self.cfg.init_state.default_joint_angles),
            joint_vel={".*": 0.0},
        )
        return EntityCfg(
            init_state=init_state,
            spec_fn=self.build_spec,
            sort_actuators=True,
            collisions=tuple(self.collision_cfgs),
            articulation=EntityArticulationInfoCfg(
                actuators=tuple(self.actuator_cfgs),
            ),
        )

