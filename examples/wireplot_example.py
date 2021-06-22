import booz_xform as bx

b = bx.Booz_xform()
b.read_wout('../tests/test_files/wout_circular_tokamak.nc')

b.run()

fig = bx.wireplot(b, js=-1)

fig.show()
