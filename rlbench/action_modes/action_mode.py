from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaIK, JointPosition
from rlbench.action_modes.gripper_action_modes import GripperActionMode, GripperJointPosition
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError('You must define your own action bounds.')


class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])


class EndEffectorActionMode(ActionMode):
    """End-effector pose control via Inverse Kinematics (Absolute Mode).

    Controls the robot by specifying the EXACT target pose (position + rotation).
    """

    def __init__(self, absolute_mode: bool = False):
        # absolute_mode=True means inputs are World Coordinates, not deltas.
        super(EndEffectorActionMode, self).__init__(
            EndEffectorPoseViaIK(absolute_mode=absolute_mode), 
            GripperJointPosition(True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        
        self.arm_action_mode.action(scene, arm_action)

        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode for Absolute Control.
        
        Structure: [X, Y, Z, Qx, Qy, Qz, Qw, Gripper]
        """
        # POS: Assuming a 1m^3 workspace centered at 0,0,0 (Adjust for your table offset!)
        # QUAT: Normalized between -1 and 1
        # GRIP: 0 to 0.04
        
        # low = np.array([
        #     -0.28, -0.66,  0.76,  # Position (X, Y, Z) - Lower limit
        #     -1.0, -1.0, -1.0, -1.0, # Quaternion (x, y, z, w)
        #      0.0                 # Gripper
        # ])
        #
        # high = np.array([
        #      0.8,  0.66,  1.76,  # Position (X, Y, Z) - Upper limit
        #      1.0,  1.0,  1.0,  1.0, # Quaternion
        #      0.04                # Gripper
        # ])
        low = np.array([
            -0.1, -0.1,  -0.1,  # Position (X, Y, Z) - Lower limit
            -0.1, -0.1, -0.1, -0.1, # Quaternion (x, y, z, w)
             0.0                 # Gripper
        ])
        
        high = np.array([
             0.1,  0.1,  0.1,  # Position (X, Y, Z) - Upper limit
             0.1,  0.1,  0.1,  0.1, # Quaternion
             0.04                # Gripper
        ])
        
        return low, high
