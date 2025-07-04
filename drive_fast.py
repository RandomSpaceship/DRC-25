import hardware
import time
import config

drive_out = hardware.MecanumHardwareAPI(
    port=config.values["hardware"]["port"],
    baudrate=config.values["hardware"]["baudrate"],
)
drive_out.open()

while True:
    # drive_out.write_speeds(400, 400)
    drive_out.update(600, -600, 600)
    time.sleep(0.05)

drive_out.close()
