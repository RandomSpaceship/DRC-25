import serial
import struct
import config
from simple_pid import PID
import time
import numpy as np
from enum import Enum
import math


class LedColors(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    RAINBOW = 3


class EGB320HardwareAPI:
    class GripperLiftTimes(Enum):
        LEVEL_1 = 30
        LEVEL_2 = 500
        LEVEL_3 = LEVEL_2 + 1150

    class GripperPositions(Enum):
        OPEN = 10
        MAX_CLOSE = 155

    def __init__(self, port=None, baudrate=115200):
        self.port = port if port else "/dev/ttyACM0"
        self.baudrate = baudrate

        WHEEL_DIA = 60  # mm
        WHEEL_BASE = 140  # mm - distance between wheels
        CLICKS_PER_REV = 20 * 2  # 20 slots * 2 for rising + falling edges
        GRIPPER_KILL_TIME = 13
        self.kill_time = GRIPPER_KILL_TIME

        wheel_circumference = math.pi * WHEEL_DIA
        self.clicks_per_mm = CLICKS_PER_REV / wheel_circumference
        turning_circumference = math.pi * WHEEL_BASE
        clicks_per_turn = self.clicks_per_mm * turning_circumference
        self.clicks_per_deg = clicks_per_turn / 720

        self.lift_time = self.GripperLiftTimes.LEVEL_1.value
        self.gripper_pos = self.GripperPositions.OPEN.value
        self.led_state = LedColors.RED.value
        self.last_gripper_time = time.monotonic()
        self.is_open = False

    def write_data(self, left, right):
        if not self.is_open:
            return
        left = int(left * 10)
        right = int(right * 10)
        can_grip = (time.monotonic() - self.last_gripper_time) > self.kill_time
        gripper_pos = self.gripper_pos if can_grip else 255
        raw_data = struct.pack(
            "<BBhhHB",
            0xAA,
            self.led_state,
            left,
            right,
            int(self.lift_time),
            int(gripper_pos),
        )

        try:
            self.serial.write(raw_data)
        except serial.PortNotOpenError as e:
            print(f"Serial port {self.port} is not open: {e}")
            self.is_open = False
            return
        except:
            pass

    # takes in a speed and a rotation in mm/s and deg/s respectively
    def update(self, drive, rotate):
        fwd_clicks = drive * self.clicks_per_mm
        turn_clicks = rotate * self.clicks_per_deg
        left = fwd_clicks - turn_clicks
        right = fwd_clicks + turn_clicks
        self.write_data(left, right)

    def set_led(self, led):
        self.led_state = led

    def set_gripper_lift_time(self, time):
        self.lift_time = time

    def set_gripper_pos(self, pos):
        p_val = pos.value
        if p_val < self.GripperPositions.OPEN.value:
            p_val = self.GripperPositions.OPEN.value
        if p_val > self.GripperPositions.MAX_CLOSE.value:
            p_val = self.GripperPositions.MAX_CLOSE.value
        if pos != self.gripper_pos:
            self.last_gripper_time = time.monotonic()
        self.gripper_pos = pos

    def open(self):
        if self.is_open:
            return
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.update(0, 0)
            self.is_open = True
        except serial.SerialException as e:
            print(f"Failed to open serial port {self.port}: {e}")
            self.is_open = False
            pass

    def close(self):
        self.serial.close()
        self.is_open = False


class MecanumHardwareAPI:
    def __init__(self, port=None, baudrate=115200):
        self.port = port if port else "/dev/ttyACM0"
        self.baudrate = baudrate

        self.is_open = False

    # Takes in drive, rotate, and strafe values in mm/s and deg/s
    def update(self, drive, rotate, strafe=0):
        if not self.is_open:
            return

        drive = int(drive)
        rotate = int(rotate)
        strafe = int(strafe)

        raw_data = struct.pack("<Bhhh", 0xAA, drive, strafe, rotate)

        try:
            self.serial.write(raw_data)
        except serial.PortNotOpenError as e:
            print(f"Serial port {self.port} is not open: {e}")
            self.is_open = False
            return
        except:
            pass

    def open(self):
        if self.is_open:
            return
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.is_open = True
        except serial.SerialException as e:
            print(f"Failed to open serial port {self.port}: {e}")
            self.is_open = False
            pass

    def close(self):
        self.serial.close()
        self.is_open = False


class HardwareAPI:
    def __init__(self):
        if config.values["hardware"]["legacy_chassis"]:
            self._hardware = EGB320HardwareAPI(
                port=config.values["hardware"]["port"],
                baudrate=config.values["hardware"]["baudrate"],
            )
        else:
            self._hardware = MecanumHardwareAPI(
                port=config.values["hardware"]["port"],
                baudrate=config.values["hardware"]["baudrate"],
            )

    def update(self, drive, rotate, strafe=0):
        if config.values["hardware"]["legacy_chassis"]:
            # no strafing on this chassis
            self._hardware.update(drive, rotate)
        else:
            self._hardware.update(drive, rotate, strafe)

    def set_led(self, led):
        self._hardware.set_led(led)

    def close(self):
        self._hardware.close()


class Controller:
    def __init__(self):
        self._hardware = HardwareAPI()
        if config.values["hardware"]["enable"]:
            self._hardware.open()

    def update(self, path_offsets):
        if not config.values["hardware"]["enable"]:
            return

        # Apply PID control
        steering_output = _steering_pid(rotate)
        strafing_output = _strafing_pid(strafe)

        self._hardware.update(drive, steering_output, strafing_output)

    def set_led(self, led):
        self._hardware.set_led(led)

    def close(self):
        self._hardware.close()


_max_turn = config.values["hardware"]["limits"]["max_turn"]
_max_strafe = config.values["hardware"]["limits"]["max_strafe"]

_steering_pid = PID(0, 0, 0, setpoint=0, output_limits=(-_max_turn, _max_turn))
_strafing_pid = PID(0, 0, 0, setpoint=0, output_limits=(-_max_strafe, _max_strafe))


def _update_coefficients():
    global _steering_pid, _strafing_pid
    pid_cfg = config.values["hardware"]["pid"]
    _steering_pid.Kp = pid_cfg["steering"]["Kp"]
    _steering_pid.Ki = pid_cfg["steering"]["Ki"]
    _steering_pid.Kd = pid_cfg["steering"]["Kd"]
    _steering_pid.reset()

    _strafing_pid.Kp = pid_cfg["strafing"]["Kp"]
    _strafing_pid.Ki = pid_cfg["strafing"]["Ki"]
    _strafing_pid.Kd = pid_cfg["strafing"]["Kd"]
    _strafing_pid.reset()


config.add_reload_handler(_update_coefficients)
