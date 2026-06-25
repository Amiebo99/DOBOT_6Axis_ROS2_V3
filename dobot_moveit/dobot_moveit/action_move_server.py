#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author FTX
@date 2025 / 03 / 03

changes made in conjunction with claude opus 4.8

Fixing ServoJ streaming
- trajectory is replayed against its own time_from_start clock,
    not firing in a tight loop and tripping moveits timeout
- resampling: moveit's sparse unevelny timed points are interpolated into a fixed control period
- servoJ 't' is matched to the stream cadence rather than a fixed 0.2s
- multithreadedexecutor, reentrantcallback and a cancel handler

tunig knobs are CONTROL_DT and SERVOJ_T
'''

import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from dobot_msgs_v3.srv import *  # noqa: F401,F403  (EnableRobot, ServoJ, ...)
import os

# tuning
CONTROL_DT = 0.05
SERVOJ_DT = 0.10

RAD2DEG = 180.0 / math.pi

def _tsec(point):
    """
    total time_from_start of a trajectory point, in seconds and using the nanoseconds too
    """
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

class FollowJointTrajectoryServer(Node):

    def __init__(self):
        super().__init__('dobot_group_controller')
        name = os.getenv("DOBOT_TYPE")
        self._cb_group = ReentrantCallbackGroup()
        
        # building followjointtrajectory
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            f'/{name}_group_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback),
            callback_group=self._cb_group,
        )
        self.get_logger().info("FollowJointTrajectory Action Server is ready...")
        
        self.EnableRobot_l = self.create_client(
            EnableRobot, '/dobot_bringup_v3/srv/EnableRobot',
            callback_group=self._cb_group)
        self.ServoJ_l = self.create_client(
            ServoJ, '/dobot_bringup_v3/srv/ServoJ',
            callback_group=self._cb_group)
       
        # must start the bringup before or this will hang
        while not self.EnableRobot_l.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
    
    def cancel_callback(self, goal_handle):
        self.get_logger().info("Cancel requested - stopping stream.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info("Received a new trajectory goal!")
        
        # Obtain the target trajectory
        trajectory = goal_handle.request.trajectory
        
        ok = self.execution_trajectory(trajectory, goal_handle)
        
        result = FollowJointTrajectory.Result()
        if not ok:                                 # mid-stream cancel
            goal_handle.canceled()
            result.error_code = -1
            return result
        
        # Return results SUCCESS
        goal_handle.succeed()
        result.error_code = 0
        return result

    def execution_trajectory(self, trajectory: JointTrajectory):
        """
        Replay the trajectory as a well-paced, interpolated servoJ stream.

        returns: True is completed, False if cancelled.
        """
        pts = trajectory.points
        if not pts:
            self.get_logger().warn("Empty Trajectory, nothing to execute.")
            return True
        total = _tsec(pts[-1])
        if total <= 0.0:
            self.get_logger().warn("Trajectory has zero duration, skipping.")
            return True

        self.get_logger().info(
            "Streaming {:.2f}s trajectory ({} pts) at {:.0f} Hz".format(
                total, len(pts), 1.0 / CONTROL_DT))
        
        t0 = time.monotonic()
        seg = 0
        tk = 0.0
        times = [_tsec(p) for p in pts]
        while tk <= total:
            if goal_handle.is_cancel_requested:
                return False
            
            # advance to the segment [pts[seg], pts[seg+1]] containing tk
            while seg < len(pts) -1 and times[seg + 1] < tk:
                seg +=1
            t_p0 = times[seg]
            t_p1 = times[seg +1] if seg + 1 < len(pts) else times[seg]
            span = t_p1 - t_p0
            alpha = 0.0 if span <= 0.0 else (tk - t_p0) / span
            alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha

            # Linear interpolation of each joint: rad -> deg
            joints_deg = [
                (p0.positions[j] + alpha * (p1.positions[j] - p0.positions[j])) * RAD2DEG
                for j in range(6)
            ]
            self.ServoJ_C(*joints_deg)

            # pace to wall clock
            tk += CONTROL_DT
            sleep = (t0 + tk) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

        # send the exact final target once more to settle on the goal
        final_deg = [p * RAD2DEG for p in pts[-1].positions]
        self.ServoJ_C(*final_deg)
        return True
        
    def ServoJ_C(self, j1, j2, j3, j4, j5, j6):  # 运动指令
        P1 = ServoJ.Request()
        P1.j1 = float(j1)
        P1.j2 = float(j2)
        P1.j3 = float(j3)
        P1.j4 = float(j4)
        P1.j5 = float(j5)
        P1.j6 = float(j6)
        P1.t = SERVOJ_T
        # optional future tuning 
        # P1.param_value = ["aheadtime=20", "gain=200"]
        self.ServoJ_l.call_async(P1)
        

def main(args=None):
    rclpy.init(args=args)
    follow_joint_trajectory_server = FollowJointTrajectoryServer()
    executor = MultiThreadedExecutor()
    executor.add_node(follow_joint_trajectory_server)
    try:
        executor.spin()
    finally:
        follow_joint_trajectory_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

    





