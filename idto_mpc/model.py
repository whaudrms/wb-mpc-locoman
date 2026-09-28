"""Build a reduced Drake model without changing the source URDF.

The contact model deliberately includes only the four foot spheres and a flat
ground. This is a locomotion model, not a self-collision or manipulation model.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pydrake.geometry import HalfSpace
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common.eigen_geometry import Quaternion
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, CoulombFriction, DiscreteContactApproximation,
)
from pydrake.systems.framework import DiagramBuilder


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "robots/b2_z1_description/urdf/b2_z1.urdf"
SRDF = ROOT / "robots/b2_z1_description/srdf/b2_z1.srdf"
FEET = tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR"))
LEGS = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
             for joint in ("hip", "thigh", "calf"))


def joint_names(arm_joints=4):
    if arm_joints not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("arm_joints must be between 0 and 6")
    return LEGS + tuple(f"joint{i}" for i in range(1, arm_joints + 1))


def model_xml(arm_joints=4):
    """Lock unused arm joints at zero, matching buildReducedRobot's neutral pose."""
    active = set(joint_names(arm_joints))
    root = ET.parse(URDF).getroot()
    for transmission in list(root.findall("transmission")):
        root.remove(transmission)
    for joint in root.findall("joint"):
        if joint.attrib["type"] != "fixed" and joint.attrib["name"] not in active:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in ("axis", "limit", "dynamics"):
                    joint.remove(child)
        # Drake does not model the URDF Coulomb joint friction attribute.
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            dynamics.attrib.pop("friction", None)
    for link in root.findall("link"):
        if link.attrib["name"] not in FEET:
            for collision in list(link.findall("collision")):
                link.remove(collision)
        # AddModelsFromString has no directory for resolving relative meshes.
        for mesh in link.findall(".//mesh"):
            mesh.set("filename", (URDF.parent / mesh.attrib["filename"]).resolve().as_uri())
    return ET.tostring(root, encoding="unicode")


def build_model(time_step, arm_joints=4, builder=None, simulation=False):
    """Return builder, plant, scene graph, and model instance (plant finalized)."""
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be positive and finite")
    builder = builder if builder is not None else DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    if simulation:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    instance = Parser(plant).AddModelsFromString(model_xml(arm_joints), "urdf")[0]
    for name in joint_names(arm_joints):
        joint = plant.GetJointByName(name, instance)
        plant.AddJointActuator(name, joint)
    plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransform(), HalfSpace(), "ground",
        CoulombFriction(0.8, 0.8),
    )
    plant.Finalize()
    expected = len(joint_names(arm_joints))
    if (plant.num_positions(), plant.num_velocities(), plant.num_actuators()) != (
            7 + expected, 6 + expected, expected):
        raise RuntimeError("Unexpected reduced model dimensions")
    return builder, plant, scene_graph, instance


class StateAdapter:
    """Pinocchio free-flyer state <-> Drake state, including reference frames.

    Pin: q=[xyz, xyzw, joints], v=[linear_B, angular_B, joints].
    Drake: floating q=[wxyz, xyz], v=[angular_W, linear_W]; joint indices are
    queried by name. Supply the actual Pinocchio model's joint name order if it
    differs from the repository's FL, FR, RL, RR, arm ordering.
    """

    def __init__(self, plant, names):
        self.plant = plant
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate joint names")
        if set(self.names) != {plant.get_joint_actuator(i).joint().name()
                               for i in self._actuator_indices()}:
            raise ValueError("Joint names must match the reduced actuated model")
        base = plant.GetBodyByName("base_link")
        self.qb = base.floating_positions_start()
        self.vb = base.floating_velocities_start_in_v()
        self.qj = np.array([plant.GetJointByName(n).position_start() for n in self.names])
        self.vj = np.array([plant.GetJointByName(n).velocity_start() for n in self.names])
        self.B = plant.MakeActuationMatrix()
        self.lower = plant.GetPositionLowerLimits()[self.qj]
        self.upper = plant.GetPositionUpperLimits()[self.qj]
        self.velocity_limits = plant.GetVelocityUpperLimits()[self.vj]
        # Retain limits even if the parser/actuator defaults change.
        joints = {j.attrib["name"]: j for j in ET.parse(URDF).getroot().findall("joint")}
        self.effort_limits = np.array([float(joints[n].find("limit").attrib["effort"])
                                       for n in self.names])

    def _actuator_indices(self):
        from pydrake.multibody.tree import JointActuatorIndex
        return [JointActuatorIndex(i) for i in range(self.plant.num_actuators())]

    @staticmethod
    def _vector(value, size, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return value.copy()

    @staticmethod
    def unit_quaternion(wxyz):
        norm = np.linalg.norm(wxyz)
        if not np.isfinite(norm) or norm < 1e-10:
            raise ValueError("Quaternion must be finite and nonzero")
        return wxyz / norm

    def to_drake(self, q_pin, v_pin):
        n = len(self.names)
        qp = self._vector(q_pin, 7 + n, "q_pin")
        vp = self._vector(v_pin, 6 + n, "v_pin")
        quat = self.unit_quaternion(qp[[6, 3, 4, 5]])
        R = RotationMatrix(Quaternion(quat)).matrix()
        q, v = np.zeros(self.plant.num_positions()), np.zeros(self.plant.num_velocities())
        q[self.qb:self.qb + 4] = quat
        q[self.qb + 4:self.qb + 7] = qp[:3]
        v[self.vb:self.vb + 3] = R @ vp[3:6]
        v[self.vb + 3:self.vb + 6] = R @ vp[:3]
        q[self.qj], v[self.vj] = qp[7:], vp[6:]
        return q, v

    def to_pin(self, q, v):
        q = self._vector(q, self.plant.num_positions(), "q_drake")
        v = self._vector(v, self.plant.num_velocities(), "v_drake")
        quat = self.unit_quaternion(q[self.qb:self.qb + 4])
        R = RotationMatrix(Quaternion(quat)).matrix()
        qp = np.r_[q[self.qb + 4:self.qb + 7], quat[[1, 2, 3, 0]], q[self.qj]]
        vp = np.r_[R.T @ v[self.vb + 3:self.vb + 6],
                   R.T @ v[self.vb:self.vb + 3], v[self.vj]]
        return qp, vp

    def actuator_torques(self, tau_pin):
        tau = self._vector(tau_pin, len(self.names), "tau_pin")
        generalized = np.zeros(self.plant.num_velocities())
        generalized[self.vj] = tau
        return self.B.T @ generalized

    def standing_state(self, context, stiffness=30000., smoothing=0.002):
        root = ET.parse(SRDF).getroot()
        pose = root.find("group_state[@name='standing_with_arm_up']")
        values = {j.attrib["name"]: np.fromstring(j.attrib["value"], sep=" ")
                  for j in pose.findall("joint")}
        qp = np.r_[values["root_joint"], [values[n][0] for n in self.names]]
        q, v = self.to_drake(qp, np.zeros(self.plant.num_velocities()))
        self.plant.SetPositions(context, q)
        foot_heights = [self.plant.EvalBodyPoseInWorld(
            context, self.plant.GetBodyByName(n)).translation()[2] - 0.032 for n in FEET]
        weight_per_foot = self.plant.CalcTotalMass(context) * 9.81 / 4
        # Static penetration for IDTO's smoothed spring contact law.
        desired_gap = -smoothing * np.log(np.expm1(weight_per_foot / stiffness / smoothing))
        q[self.qb + 6] += desired_gap - min(foot_heights)
        self.plant.SetPositions(context, q)
        return q, v
