import numpy as np
import matplotlib.pyplot as plt
import booz_xform as bx


b = bx.Booz_xform()
#  b.verbose = 0

#  b.read_wout('./data/wout_ncsx_c09r00_free.nc')

b.read_wout('../tests/test_files/wout_up_down_asymmetric_tokamak.nc')

b.run()

s = np.linspace(0, b.psi_lcfs)

fig, axs = plt.subplots(2,2,tight_layout=True)

axs = axs.ravel()

axs[0].plot(s, b.g(s))
axs[0].plot(b.psi_b, b.Boozer_G, 'r+')
axs[0].set_title("G")

axs[1].plot(s, b.I(s))
axs[1].plot(b.psi_b, b.Boozer_I, 'r+')
axs[1].set_title("I")

axs[2].plot(s, b.iota_m(s))
axs[2].plot(b.psi_in, b.iota, 'r+')
axs[2].set_title("iota")

axs[3].plot(s, b.q(s))
axs[3].plot(b.psi_in, 1/b.iota, 'r+')
axs[3].set_title("q")
plt.show()
