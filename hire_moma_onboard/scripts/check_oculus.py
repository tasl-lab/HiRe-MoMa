"""Print Quest 3 controller poses and buttons (connection check).

  python hire_moma_onboard/scripts/check_oculus.py --ip <quest-ip>     # omit --ip for USB
"""
import argparse
import time

from oculus_reader import OculusReader

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default=None)
args = parser.parse_args()
reader = OculusReader(ip_address=args.ip)
while True:
    time.sleep(0.3)
    print(reader.get_transformations_and_buttons())
