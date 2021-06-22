import time
import numpy as np
import matplotlib.pyplot as plt
import booz_xform as bx


def calculate_B_boozer_on_surface_vectorized(b, js, phi, theta):

    input_shape = phi.shape
    #  assert theta.shape == input_shape

    m = b.xm_b
    n = b.xn_b

    angle = m * theta[..., np.newaxis] - n * phi[..., np.newaxis]

    print(angle.shape)

    modB = np.cos(angle) @ b.bmnc_b[:, js]

    if b.asym:
        modB += np.sin(angle) @ b.bmns_b[:, js]

    return modB.reshape(input_shape)


b = bx.Booz_xform()
#  b.verbose = 0

#  b.read_wout('./data/wout_ncsx_c09r00_free.nc')

b.read_wout('../tests/test_files/wout_LandremanSenguptaPlunk_section5p3.nc')

b.run()

ntheta = 500
nphi = 90

theta1d = np.linspace(0, 2 * np.pi, ntheta)
phi1d = np.linspace(0, 2 * np.pi / b.nfp, nphi)
phi, theta = np.meshgrid(phi1d, theta1d)

B = b.calculate_modB_boozer_on_surface(js=-1, phi=phi, theta=theta)

start = time.time()
B = b.calculate_modB_boozer_on_surface(js=-1, phi=phi, theta=theta)
end = time.time()
print("Elapsed (not vectorized) = %s" % (end - start))

start = time.time()
B_vectorized = calculate_B_boozer_on_surface_vectorized(b, js=-1, phi=phi, theta=theta)
end = time.time()
print("Elapsed (vectorized) = %s" % (end - start))
bx.surfplot(b, js=-1)


start = time.time()
B_model = b.mod_B_model()

dB_dtheta = b.calculate_modB_boozer_deriv_on_surface(js=-1, phi=phi, theta=theta, phi_order=0, theta_order=1)

dB_dtheta_model = B_model.deriv(r_order=0, theta_order=1, phi_order=0)

dB_dtheta_modelled = dB_dtheta_model.calculate_on_surface(b.s_in[-1],
                                                          theta,
                                                          phi)


dB2_dtheta2_model1 = B_model.deriv(r_order=0, theta_order=2, phi_order=0)
dB2_dtheta2_model2 = dB_dtheta_model.deriv(r_order=0, theta_order=1, phi_order=0)


dB2_dtheta2_modelled_1 = dB2_dtheta2_model1.calculate_on_surface(b.s_in[-1],
                                                                 theta,
                                                                 phi)

dB2_dtheta2_modelled_2 = dB2_dtheta2_model2.calculate_on_surface(b.s_in[-1],
                                                                 theta,
                                                                 phi)
end = time.time()
print("Elapsed (models) = %s" % (end - start))
fig, ax = plt.subplots()

ax.contour(phi, theta, dB_dtheta_modelled, [0, ])
ax.contour(phi, theta, dB_dtheta, [0, ], colors='red')
plt.show()
