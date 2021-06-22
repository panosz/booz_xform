R"""
This script demonstrates a small discrepancy in the calculation of <|B|>
when it is performed in boozer and vmec angles.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import booz_xform as bx



b = bx.Booz_xform()
b.verbose = 0
b.read_wout('../tests/test_files/wout_LandremanSenguptaPlunk_section5p3.nc')

b.run()


# calculate the average value of the magnetic field by taking the (0,0) component
# of the cos coefficients in boozer angles

mean_b_booz = b.bmnc_b[np.logical_and(b.xm_b == 0, b.xn_b == 0), :]

# do the same in vmec angles
mean_b_vmec = b.bmnc[np.logical_and(b.xm_nyq == 0, b.xn_nyq == 0), :]


print("Discrepancy of the two methods:")
print(np.abs(mean_b_vmec - mean_b_booz).max())

