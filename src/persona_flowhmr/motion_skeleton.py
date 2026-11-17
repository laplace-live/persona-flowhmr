"""Reference VRM T-pose for animation-only files. Local translations in meters,
in SMPL-H joint order and canonical (+X left, +Y up, +Z forward) coordinates.
"""

# A representative 1.68 m rig with the pelvis at 56.7% of standing height.
# These rest distances describe the source rig for retargeting, independently
# of the recorded performer's motions or the avatar chosen by the player.
REFERENCE_HEIGHT_M = 1.68
REST_HIPS_HEIGHT_M = REFERENCE_HEIGHT_M * 0.567
_BODY = (
    (0, REST_HIPS_HEIGHT_M, 0),                    # hips
    (0.09, -0.04, 0), (-0.09, -0.04, 0),           # upper legs
    (0, 0.11, 0),                                  # spine
    (0, -0.43, 0), (0, -0.43, 0),                  # lower legs
    (0, 0.12, 0),                                  # chest
    (0, -0.41, 0), (0, -0.41, 0),                  # ankles
    (0, 0.13, 0),                                  # upper chest
    (0, -0.06, 0.14), (0, -0.06, 0.14),           # toes
    (0, 0.16, 0),                                  # neck
    (0.045, 0.09, 0), (-0.045, 0.09, 0),           # shoulders
    (0, 0.095, 0),                                 # head
    (0.13, 0, 0), (-0.13, 0, 0),                   # upper arms
    (0.275, 0, 0), (-0.275, 0, 0),                 # elbows
    (0.25, 0, 0), (-0.25, 0, 0),                   # wrists
)
_LEFT_FINGERS = (
    (0.080, 0, 0.027), (0.040, 0, 0), (0.024, 0, 0),   # index
    (0.085, 0, 0.009), (0.043, 0, 0), (0.026, 0, 0),   # middle
    (0.066, 0, -0.029), (0.030, 0, 0), (0.020, 0, 0),  # little
    (0.080, 0, -0.011), (0.038, 0, 0), (0.024, 0, 0),  # ring
    (0.024, 0, 0.024), (0.027, 0, 0.027), (0.020, 0, 0.020),  # thumb
)
REST_OFFSETS_M = _BODY + _LEFT_FINGERS + tuple((-x, y, z) for x, y, z in _LEFT_FINGERS)
