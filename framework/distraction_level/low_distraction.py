# # framework/distraction_level/low_distraction.py
#
# def show_indicator(active: bool):
#     """
#     When active=True → print a warning message.
#     When active=False → print nothing.
#     """
#     if active:
#         print("⚠️  PLEASE PAY ATTENTION TO THE ROAD  ⚠️")

import os
import cv2

_WINDOW_NAME = "Driver Dashboard"

_dashboard_safe = None
_dashboard_warning = None

def _load_images():

    global _dashboard_safe, _dashboard_warning

    if _dashboard_safe is not None and _dashboard_warning is not None:
        return

    base_dir = os.path.dirname(__file__)
    safe_path = os.path.join(base_dir, "dashboard_safe.png")
    warn_path = os.path.join(base_dir, "dashboard_warning.png")

    _dashboard_safe = cv2.imread(safe_path)
    _dashboard_warning = cv2.imread(warn_path)

def show_indicator(active: bool):

    _load_images()
    if _dashboard_safe is None or _dashboard_warning is None:
        return

    img = _dashboard_warning if active else _dashboard_safe

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(_WINDOW_NAME, 830, 525)
    cv2.resizeWindow(_WINDOW_NAME, 875, 175)
    cv2.imshow(_WINDOW_NAME, img)
    cv2.waitKey(1)
