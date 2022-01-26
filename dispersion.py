import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import plasma_dispersion as pd

vt_c = 0.3
sq2 = 2 ** 0.5


def electrostatic_dispersion(k, z, vb):
    k_sq = k ** 2.0
    z_e_one = (z - vb) / sq2
    z_e_two = (z + vb) / sq2

    return 1 - 0.5 * (pd.Zprime(z_e_one) + pd.Zprime(z_e_two)) / k_sq


def dispersion_function(k, z, vb):
    """
    Computes plasma dispersion function epsilon_perp(k, zeta) = 0 for two beams
    """
    k_sq = k ** 2.0
    z_e = z / sq2

    return 1 - (z * vt_c) ** 2.0 + (1 + 0.5 * (1 + vb ** 2.0) * pd.Zprime(z_e)) / k_sq * (vt_c ** 2.0)


def dispersion_function_aniso(k, z, vty_vtx):
    """
    Computes plasma dispersion function epsilon_perp(k, zeta) = 0 for anisotropic maxwellian
    """
    k_sq = k ** 2.0
    z_e = z / sq2

    return 1 - (z * vt_c) ** 2.0 + (1 + 0.5 * (vty_vtx ** 2.0) * pd.Zprime(z_e)) / k_sq * (vt_c ** 2.0)


def analytic_jacobian(k, z, vb):
    k_sq = k ** 2.0
    z_e = z / sq2

    return (2.0 * z + 0.5 * (1 + vb ** 2.0) * pd.Zdoubleprime(z_e) / k_sq) * (vt_c ** 2.0)


def dispersion_fsolve(z, k, vb):
    complex_z = z[0] + 1j * z[1]
    d = dispersion_function(k, complex_z, vb)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(z, k, vb):
    complex_z = z[0] + 1j * z[1]
    jac = analytic_jacobian(k, complex_z, vb)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


# Phase velocities
zr = np.linspace(-4, 4, num=200)
zi = np.linspace(-4, 4, num=200)
z = np.tensordot(zr, np.ones_like(zi), axes=0) + 1.0j * np.tensordot(np.ones_like(zr), zi, axes=0)

ZR, ZI = np.meshgrid(zr, zi, indexing='ij')

wavenumber = 0.1
mu = dispersion_function_aniso(wavenumber, z, 3)
# ep = electrostatic_dispersion(wavenumber, z, 1)

solution = opt.root(dispersion_fsolve, x0=np.array([0, 0.5]),
                    args=(wavenumber, 1), jac=jacobian_fsolve, tol=1.0e-15)
print(solution.x[1])

# plt.figure()
# plt.contour(ZR, ZI, np.real(ep), 0, colors='r', linewidths=3)
# plt.contour(ZR, ZI, np.imag(ep), 0, colors='g', linewidths=3)
# plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
# plt.grid(True), plt.title('Static, longitudinal'), plt.tight_layout()

plt.figure()
plt.contour(ZR, ZI, np.real(mu), 0, colors='r', linewidths=3)
plt.contour(ZR, ZI, np.imag(mu), 0, colors='g', linewidths=3)
plt.xlabel('Real phase velocity'), plt.ylabel('Imaginary phase velocity')
plt.grid(True), plt.title('Dynamic, transverse'), plt.tight_layout()

plt.show()

# Obtain some solutions
k = np.linspace(0.001, np.pi, num=12000)
sols = np.zeros_like(k) + 0j
guess_r, guess_i = 0.1, 0.612
for idx, wave in enumerate(k):
    guess_r += 2e-1
    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                        args=(wave, 1), jac=jacobian_fsolve, tol=1.0e-10)
    guess_r, guess_i = solution.x
    sols[idx] = (guess_r + 1j * guess_i)

skin_depth_single_beam = 0.3 / np.sqrt(2)

plt.figure()
plt.plot(k, np.real(sols), 'r', linewidth=3, label='Real')
plt.plot(k, np.imag(sols), 'g', linewidth=3, label='Imaginary')
plt.plot(k, 10 / 3 * np.ones_like(k), 'k--', label='Lightspeed', linewidth=3)
plt.plot([skin_depth_single_beam, skin_depth_single_beam], [0, 10 / 3], 'b--', linewidth=1,
         label='Individual skin depth')
plt.xlabel(r'Wavenumber $k\lambda_D$'), plt.ylabel(r'Phase velocity $\zeta / v_t$')
plt.title(r'Weibel instability | Beam $v_b/c = 0.3$, Thermal $v_t/c = 0.3$')
plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

# plt.figure()
# plt.plot(k, k * np.real(sols), 'r', label='real', linewidth=3)
# plt.plot(k, k * np.imag(sols), 'g', label='imag', linewidth=3)
# plt.xlabel(r'Wavenumber k'), plt.ylabel(r'Frequency $\omega_p / \omega_{pe}$')
# plt.grid(True), plt.legend(loc='best'), plt.tight_layout()
plt.show()
