import numpy as np

N_BODYS = 32
SPACE_SIZE = 1e20 # meters
MAX_SPEED = 1e5 # m/s
SOLAR_MASS = 2e30 # kg
MIN_MASS = 0.1 * SOLAR_MASS
MAX_MASS = 1.2 * SOLAR_MASS


masses = np.abs(np.random.normal((MAX_MASS + MIN_MASS) / 2.0, (MAX_MASS - MIN_MASS) / 2.0, N_BODYS))
x0 = np.random.rand(N_BODYS) * SPACE_SIZE
y0 = np.random.rand(N_BODYS) * SPACE_SIZE
z0 = np.random.rand(N_BODYS) * SPACE_SIZE

vx0 = np.random.rand(N_BODYS) * MAX_SPEED
vy0 = np.random.rand(N_BODYS) * MAX_SPEED
vz0 = np.random.rand(N_BODYS) * MAX_SPEED

out = open("catalogue32.csv", "w")

for i in range(N_BODYS):
 s = "%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n" % (masses[i], x0[i], y0[i], z0[i], vx0[i], vy0[i], vz0[i])
 out.write(s)
 
out.close()
