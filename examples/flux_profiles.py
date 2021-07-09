import numpy as np
import matplotlib.pyplot as plt
import booz_xform as bx
import netCDF4 as nc


b = bx.Booz_xform()

#  wout_file = '../tests/test_files/wout_up_down_asymmetric_tokamak.nc'
wout_file = '../tests/test_files/wout_li383_1.4m.nc'

b.read_wout(wout_file)

ds = nc.Dataset(wout_file)
phi = ds.variables['phi'][:].data

s_f = np.linspace(0, 1, num=ds['ns'][:].data)

dphi_dsf = ds.variables['phipf'][:].data
psi_pf = ds.variables['chi'][:].data

b.run()

psi_in = s_f * b.psi_lcfs
psi_i = np.linspace(0, b.psi_lcfs)

fig, axs = plt.subplots(2,3,tight_layout=True)

axs = axs.ravel()

axs[0].plot(psi_i, b.g(psi_i))
axs[0].plot(b.psi_b, b.Boozer_G, 'r+')
axs[0].set_title("G")

axs[1].plot(psi_i, b.I(psi_i))
axs[1].plot(b.psi_b, b.Boozer_I, 'r+')
axs[1].set_title("I")

axs[2].plot(psi_i, b.iota_m(psi_i))
axs[2].plot(b.psi_in, b.iota, 'r+')
axs[2].set_title("iota")

axs[3].plot(psi_i, b.q(psi_i))
axs[3].plot(b.psi_in, 1/b.iota, 'r+')
axs[3].set_title("q")

axs[4].plot(psi_in, - psi_pf, 'r+')
axs[4].plot(psi_i, b.psi_p(psi_i))

plt.show()
