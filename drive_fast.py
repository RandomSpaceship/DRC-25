import hardware
import time
import config

drive_out = hardware.MecanumHardwareAPI(
    port=config.values["hardware"]["port"],
    baudrate=config.values["hardware"]["baudrate"],
)
drive_out.open()

for i in range(0, 50):
    # drive_out.write_speeds(400, 400)
    drive_out.update(-300, 0, 0)
    time.sleep(0.05)

drive_out.close()
