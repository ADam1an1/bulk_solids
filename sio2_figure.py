"""Explanatory figures for the quartz inputs in insulator_bandgap.ipynb, matching si_figure.py.

Panels
  A  The relaxed quartz structure, built as QE builds it from ibrav=4, celldm(1), celldm(3), ATOMIC_POSITIONS
  B  Each hard-coded input constant -> what it means geometrically / physically
  C  Reciprocal space: the G-vectors allowed by the lattice and the two cutoff spheres
  D  Why a cutoff works: a periodic function rebuilt from plane waves up to increasing cutoff

Every setting is read from the final SCF input, and the numbers computed from it are checked
against that run's pw.x output.
"""
from __future__ import annotations

import re
from itertools import combinations, product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from si_figure import AQUA, BLUE, BOHR_A, GRID, INK, INK2, ORANGE, SEQ

SI_O_BOND_MAX = 1.8                                    # Å: longer than any Si-O bond, shorter than any other pair
FORMS = {
    "alpha": dict(name="α-quartz", space_group="P3₂21", symmetry_ops=6, experiment=(4.9137, 5.4047)),
    "beta": dict(name="β-quartz", space_group="P6₂22", symmetry_ops=12, experiment=(4.9965, 5.4546)),
}


def read_input(which, text):
    """The form's fixed details, plus every setting read from a pw.x input, so the figure can't drift from the run."""
    value = lambda key: float(re.search(rf"{re.escape(key)}\s*=\s*([0-9.]+)", text)[1])
    block = re.search(r"ATOMIC_POSITIONS \(crystal\)\s*\n(.*?)\n\s*\n", text, re.S)[1]
    positions = [(row.split()[0], *map(float, row.split()[1:4])) for row in block.strip().splitlines()]
    mesh = tuple(int(n) for n in re.search(r"K_POINTS \(automatic\)\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text).groups())
    return {**FORMS[which], "celldm1": value("celldm(1)"), "celldm3": value("celldm(3)"), "ecutwfc": value("ecutwfc"),
            "ecutrho": value("ecutrho"), "nbnd": int(value("nbnd")), "mesh": mesh, "positions": positions}


def lattice(form):
    """QE's ibrav=4 primitive vectors (rows, units of a) and reciprocal vectors (rows, units of 2*pi/a)."""
    c_over_a = form["celldm3"]
    a_prim = np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, c_over_a]])
    return a_prim, np.linalg.inv(a_prim).T


def fractions(form):
    return np.array([p[1:] for p in form["positions"]]), np.array([p[0] for p in form["positions"]])


def irreducible_kpoints(form):
    """Symmetry-reduced k-points of an unshifted Monkhorst-Pack mesh (Cartesian, 2pi/a units) via spglib."""
    import spglib
    a_prim, b_rec = lattice(form)
    frac, species = fractions(form)
    cell = (a_prim, frac, [14 if s == "Si" else 8 for s in species])
    mapping, grid = spglib.get_ir_reciprocal_mesh(form["mesh"], cell, is_shift=[0, 0, 0], symprec=1e-4)
    return (grid[np.unique(mapping)] / np.array(form["mesh"])) @ b_rec


def g_vectors(form, nmax=None):
    if nmax is None:                                   # n_i = G.a_i, so |n_i| can't exceed |G| |a_i|
        radius = np.sqrt(form["ecutrho"]) * form["celldm1"] / (2 * np.pi)
        nmax = [int(radius * np.linalg.norm(v)) + 2 for v in lattice(form)[0]]
    n = np.array(list(product(*(range(-m, m + 1) for m in nmax))), float)
    return n @ lattice(form)[1]


def plane_wave_counts(form, kpts):
    """Plane waves per k-point, and the G-vectors on QE's dense (ecutrho) and smooth (4*ecutwfc) grids."""
    tpiba2 = (2 * np.pi / form["celldm1"]) ** 2         # Ry per (2pi/a)^2, since E[Ry] = |q|^2 [bohr^-2]
    G = g_vectors(form)
    g2 = (G ** 2).sum(1) * tpiba2
    near = G[g2 < 2 * form['ecutwfc']]
    npw_k = [int((((near + k) ** 2).sum(1) * tpiba2 <= form['ecutwfc']).sum()) for k in kpts]
    return dict(tpiba2=tpiba2, npw_k=npw_k, n_rho=int((g2 <= form['ecutrho']).sum()), n_smooth=int((g2 <= 4 * form['ecutwfc']).sum()))


def bonds(form):
    """All Si-O bonds touching an atom in the cell, in Cartesian units of a, with bond lengths and Si-O-Si angles."""
    a_prim, _ = lattice(form)
    frac, species = fractions(form)
    a_ang = form["celldm1"] * BOHR_A
    silicon, oxygen = np.nonzero(species == "Si")[0], np.nonzero(species == "O")[0]
    close = lambda vector: np.linalg.norm(vector) * a_ang < SI_O_BOND_MAX
    shifts = list(product((-1, 0, 1), repeat=3))

    tetrahedra = {i: [(frac[j] + s) @ a_prim for j in oxygen for s in shifts if close((frac[j] + s - frac[i]) @ a_prim)]
                  for i in silicon}
    assert all(len(o) == 4 for o in tetrahedra.values())
    lengths = np.array([np.linalg.norm(o - frac[i] @ a_prim) * a_ang for i, group in tetrahedra.items() for o in group])

    links, angles = [], []                              # the two Si bonded to each O in the cell
    for j in oxygen:
        here = frac[j] @ a_prim
        partners = [((frac[i] + s) @ a_prim, s == (0, 0, 0)) for i in silicon for s in shifts
                    if close((frac[i] + s - frac[j]) @ a_prim)]
        assert len(partners) == 2, (j, len(partners))
        links += [(here, si, inside) for si, inside in partners]
        v, w = (si - here for si, _ in partners)
        angles.append(np.degrees(np.arccos(v @ w / np.linalg.norm(v) / np.linalg.norm(w))))
    return tetrahedra, links, lengths, np.array(angles)


def _structure_panel(ax, form):
    a_prim, _ = lattice(form)
    frac, species = fractions(form)
    tetrahedra, links, lengths, angles = bonds(form)
    a_ang = form["celldm1"] * BOHR_A
    a1, a2, a3 = a_prim
    cart = frac @ a_prim

    corners = [np.zeros(3), a1, a1 + a2, a2]
    for lower, upper in zip(corners, corners[1:] + corners[:1]):
        for z in (np.zeros(3), a3):
            ax.plot(*zip(lower + z, upper + z), color=INK2, lw=0.8, alpha=0.6)
    for corner in corners:
        ax.plot(*zip(corner, corner + a3), color=INK2, lw=0.8, alpha=0.6)

    faces = [list(face) for group in tetrahedra.values() for face in combinations(group, 3)]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=BLUE, edgecolor="none", alpha=0.10))
    for i, group in tetrahedra.items():
        for o in group:
            ax.plot(*zip(cart[i], o), color="#9a9890", lw=2.0, solid_capstyle="round", zorder=1)
    for o, si, _ in links:
        ax.plot(*zip(o, si), color="#9a9890", lw=2.0, solid_capstyle="round", zorder=1)

    in_cell = cart[species == "O"]
    o_images = np.array([o for group in tetrahedra.values() for o in group
                         if np.abs(in_cell - o).sum(1).min() > 1e-6])
    si_images = np.array([si for _, si, inside in links if not inside])
    ax.scatter(*o_images.T, s=42, facecolor="white", edgecolor=ORANGE, linewidth=1.5, depthshade=False, zorder=3)
    ax.scatter(*si_images.T, s=70, facecolor="white", edgecolor=BLUE, linewidth=1.5, depthshade=False, zorder=3)
    ax.scatter(*in_cell.T, s=90, color=ORANGE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=4)
    ax.scatter(*cart[species == "Si"].T, s=190, color=BLUE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=5)

    for i, v in enumerate(a_prim):
        ax.quiver(0, 0, 0, *v, color=INK, lw=1.8, arrow_length_ratio=0.08, zorder=6)
        ax.text(*(v * 1.06 + np.array([0.02, 0, 0.02])), f"a$_{i + 1}$", color=INK, fontsize=10, weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    bond_text = (f"Si–O bonds {lengths.min():.3f}–{lengths.max():.3f} Å" if np.ptp(lengths) > 5e-4
                 else f"Si–O bond {lengths.mean():.3f} Å")
    leg = [Line2D([], [], ls="", marker="o", ms=11, mfc=BLUE, mec="white", label="Si: 3 in the cell"),
           Line2D([], [], ls="", marker="o", ms=8, mfc=ORANGE, mec="white", label="O: 6 in the cell"),
           Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=BLUE, mew=1.5, label="Si or O image (hollow) in a neighbouring cell"),
           Line2D([], [], color=INK, lw=1.8, label="primitive vectors from ibrav = 4"),
           Line2D([], [], color="#9a9890", lw=2.2, label=bond_text)]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(-0.02, 0.99), fontsize=8, frameon=False)
    ax.text2D(0.0, -0.08, f"a = celldm(1) = {form['celldm1']:.6f} bohr = {a_ang:.4f} Å,  "
              f"c = celldm(3)·a = {form['celldm3'] * a_ang:.4f} Å\n"
              f"shaded: SiO₄ tetrahedra around the three Si; each O joins two Si at {angles.mean():.1f}°",
              transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(-0.75, 1.15); ax.set_ylim(-0.3, 1.2); ax.set_zlim(-0.3, 1.5)
    ax.set_box_aspect((1.9, 1.5, 1.8), zoom=1.12)        # proportional to the axis ranges, so the cell isn't distorted
    ax.view_init(elev=14, azim=-58)
    ax.set_axis_off()
    ax.set_title(f"A   {form['name']} as relaxed, for the band calculation", loc="left", fontsize=11, weight="bold", color=INK)


def _mapping_panel(ax, form, counts, volume):
    ax.set_axis_off()
    ax.set_title("B   What each hard-coded constant sets", loc="left", fontsize=11, weight="bold", color=INK)
    a_ang = form["celldm1"] * BOHR_A
    c_ang = form["celldm3"] * a_ang
    change = lambda value, i: f"{100 * (value / form['experiment'][i] - 1):+.2f}%"
    electrons = 3 * 4 + 6 * 6
    grid = form["mesh"]
    rows = [
        ("ibrav = 4", "Hexagonal Bravais lattice. QE generates", "a₁ = a(1,0,0)  a₂ = a(−½,√3/2,0)  a₃ = c(0,0,1)"),
        (f"celldm(1) = {form['celldm1']:.6f}", f"a, relaxed: {a_ang:.4f} Å ({change(a_ang, 0)} vs experiment)",
         f"cell volume (√3/2)·a²c = {volume:.1f} bohr³"),
        (f"celldm(3) = {form['celldm3']:.6f}", f"c/a, relaxed: c = {c_ang:.4f} Å ({change(c_ang, 1)} vs experiment)",
         "hexagonal cells need this second length"),
        ("nat = 9", "3 Si + 6 O: three SiO₂ units per cell",
         f"{form['space_group']}: pw.x finds {form['symmetry_ops']} symmetry operations"),
        ("ntyp = 2", "Si and O, both PBE ultrasoft (rrkjus)", "4 valence e⁻ per Si, 6 per O"),
        ("ATOMIC_POSITIONS (crystal)", "relaxed fractions of a₁, a₂, a₃", "copied unrounded from vc-relax, to keep the symmetry"),
        ("nbnd = 32", f"3×4 + 6×6 = {electrons} e⁻ → {electrons // 2} filled bands", f"+{form['nbnd'] - electrons // 2} empty bands, so the gap can be read off"),
        (f"K_POINTS {grid[0]} {grid[1]} {grid[2]} 0 0 0", f"{grid[0]}×{grid[1]}×{grid[2]} = {np.prod(grid)} k-points for the SCF density",
         f"→ {len(counts['npw_k'])} after symmetry; the band path reuses this density"),
        (f"ecutwfc = {form['ecutwfc']:g} Ry", f"ψ basis: plane waves with |k+G|² ≤ {form['ecutwfc']:g} Ry",
         f"→ {min(counts['npw_k'])}–{max(counts['npw_k'])} plane waves per k-point"),
        (f"ecutrho = {form['ecutrho']:g} Ry", f"{form['ecutrho'] / form['ecutwfc']:g} × ecutwfc, for the sharp ultrasoft charges in ρ",
         f"→ {counts['n_rho']} G-vectors ({counts['n_smooth']} for the smooth part)"),
    ]
    y, step = 0.96, 0.096
    for key, l1, l2 in rows:
        ax.text(0.0, y, key, family="monospace", fontsize=9, weight="bold", color=INK, va="top", transform=ax.transAxes)
        ax.text(0.40, y, l1, fontsize=8.8, color=INK, va="top", transform=ax.transAxes)
        ax.text(0.40, y - 0.04, l2, fontsize=8.3, color=INK2, va="top", transform=ax.transAxes)
        ax.plot([0, 1], [y - 0.08, y - 0.08], color=GRID, lw=0.8, transform=ax.transAxes)
        y -= step


def _reciprocal_panel(ax, form, counts):
    tpiba2 = counts["tpiba2"]
    r_wf, r_rho = np.sqrt(form['ecutwfc'] / tpiba2), np.sqrt(form['ecutrho'] / tpiba2)
    G = g_vectors(form, nmax=(48, 48, 0))               # basal plane through Gamma
    u, v = G[:, 0], G[:, 1]
    g2 = (G ** 2).sum(1)
    lim = r_rho * 1.12
    keep = (abs(u) <= lim) & (abs(v) <= lim)
    in_wf, in_rho = g2 <= r_wf ** 2, (g2 > r_wf ** 2) & (g2 <= r_rho ** 2)
    out = ~(in_wf | in_rho)
    ax.scatter(u[keep & out], v[keep & out], s=3, color="#c9c7bf", lw=0)
    ax.scatter(u[keep & in_rho], v[keep & in_rho], s=5, color=AQUA, lw=0)
    ax.scatter(u[keep & in_wf], v[keep & in_wf], s=7, color=BLUE, lw=0)
    ax.add_patch(Circle((0, 0), r_rho, fill=False, color=AQUA, lw=2))
    ax.add_patch(Circle((0, 0), r_wf, fill=False, color=BLUE, lw=2))
    ax.annotate(f"|k+G|² ≤ ecutwfc = {form['ecutwfc']:g} Ry\n→ ψ uses {min(counts['npw_k'])}–{max(counts['npw_k'])} plane waves",
                (r_wf * 0.707, r_wf * 0.707), xytext=(r_wf * 0.7, r_rho * 0.62), fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    ax.annotate(f"|G|² ≤ ecutrho = {form['ecutrho'] / form['ecutwfc']:g}·ecutwfc = {form['ecutrho']:g} Ry\n→ ρ uses {counts['n_rho']} G-vectors",
                (-r_rho * 0.707, r_rho * 0.707), xytext=(-lim * 0.97, lim * 0.97), fontsize=8.5, color=INK,
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=AQUA, lw=1),
                arrowprops=dict(arrowstyle="-", color=AQUA, lw=1))
    ax.text(lim * 0.97, -lim * 0.97, "grey: excluded (above both cutoffs)", fontsize=8, color=INK2, ha="right",
            va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.scatter([0], [0], s=40, color=INK, zorder=5)
    ax.text(0.8, -0.8, "Γ", fontsize=10, color=INK, ha="left", va="top", weight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("G along x   (units of 2π/a)", fontsize=8.5, color=INK2)
    ax.set_ylabel("G along y   (units of 2π/a)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("C   Crystal → plane-wave basis (reciprocal lattice, basal plane through Γ)", loc="left",
                 fontsize=11, weight="bold", color=INK, pad=44)
    ax.text(0.0, 1.0, r"$\psi_{n\mathbf{k}}(\mathbf{r})=\sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,"
                      r"e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$,   $\mathbf{G}=n_1\mathbf{b}_1+n_2\mathbf{b}_2+n_3\mathbf{b}_3$"
                      "\nb₁ = 2π/a(1,1/√3,0)  b₂ = 2π/a(0,2/√3,0)  b₃ = 2π/c(0,0,1)  (hexagonal; set by ibrav and celldm)",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")


def _cutoff_panel(ax, form):
    # 1-D illustration along a straightened Si-O-Si chain: period is two mean Si-O bonds
    L = 2 * bonds(form)[2].mean() / BOHR_A
    x = np.linspace(0, L, 1200)
    features = ((0.0, 0.55, 0.6), (0.5, 0.35, 1.0))      # centre, width in bohr, height: chosen for illustration
    m = np.arange(-200, 201)
    g = 2 * np.pi * m / L                                # bohr^-1 ; E = g^2 in Ry
    coef = sum(h * s * np.exp(-0.5 * (s * g) ** 2) * np.exp(-1j * g * c * L) for c, s, h in features) * np.sqrt(2 * np.pi) / L
    f = lambda emax: np.real(np.exp(1j * np.outer(x, g[g ** 2 <= emax])) @ coef[g ** 2 <= emax])
    for emax, color in zip([10, 30, form['ecutwfc']], SEQ):
        n = int((g ** 2 <= emax).sum())
        ax.plot(x / L, f(emax), color=color, lw=2, label=f"cutoff {emax:g} Ry  ({n} plane waves)")
    ax.plot(x / L, f(1e9), color=INK, lw=1.4, ls=(0, (4, 3)), label="target function")
    for c, label, color in [(0, "Si", BLUE), (0.5, "O", ORANGE), (1, "Si", BLUE)]:
        ax.scatter([c], [-0.24], s=70, color=color, edgecolor="white", lw=1.5, zorder=5)
        ax.text(c + (0.015 if c < 1 else -0.015), -0.24, label, ha="left" if c < 1 else "right", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.32, 1.12)
    ax.set_xlabel("position along a straightened Si–O–Si chain   (units of two Si–O bonds)", fontsize=8.5, color=INK2)
    ax.set_ylabel("amplitude (arb.)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.0))
    ax.set_title("D   What the cutoff does (1-D sketch, not calculated data)", loc="left",
                 fontsize=11, weight="bold", color=INK, pad=44)
    ax.text(0.0, 1.0, "Sharper features need higher-energy plane waves, and oxygen's electrons are held more tightly than silicon's.\n"
                      f"Peak widths and heights are illustrative. The {form['ecutwfc']:g} Ry value itself comes from the convergence tests.",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")


def compare_with_output(form, counts, volume, text):
    """Compare the computed numbers with pw.x output for the same input. Returns any differences."""
    found = {"k-points": (int(re.search(r"number of k points=\s+(\d+)", text)[1]), len(counts["npw_k"])),
             "dense G-vectors": (int(re.search(r"Dense\s+grid:\s+(\d+) G-vectors", text)[1]), counts["n_rho"]),
             "smooth G-vectors": (int(re.search(r"Smooth grid:\s+(\d+) G-vectors", text)[1]), counts["n_smooth"]),
             "symmetry operations": (int(re.search(r"(\d+) Sym\. Ops", text)[1]), form["symmetry_ops"])}
    plane_waves = sorted(int(n) for n in re.findall(r"\(\s*(\d+) PWs\)", text)[-len(counts["npw_k"]):])
    found["plane waves"] = (plane_waves, sorted(counts["npw_k"]))
    differences = {key: pair for key, pair in found.items() if pair[0] != pair[1]}
    pw_volume = float(re.search(r"unit-cell volume\s+=\s+([\d.]+)", text)[1])
    if abs(pw_volume - volume) > 1e-2:
        differences["volume"] = (pw_volume, volume)
    return differences


def make_sio2_figure(which, scf_input, scf_output):
    """Figure for one form of quartz, from the text of its final SCF input and output."""
    form = read_input(which, scf_input)
    counts = plane_wave_counts(form, irreducible_kpoints(form))
    volume = abs(np.linalg.det(lattice(form)[0])) * form["celldm1"] ** 3
    counts["differences"] = compare_with_output(form, counts, volume, scf_output)

    fig = plt.figure(figsize=(15, 12.5), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1], hspace=0.32, wspace=0.12,
                          left=0.04, right=0.98, top=0.95, bottom=0.06)
    _structure_panel(fig.add_subplot(gs[0, 0], projection="3d"), form)
    _mapping_panel(fig.add_subplot(gs[0, 1]), form, counts, volume)
    _reciprocal_panel(fig.add_subplot(gs[1, 0]), form, counts)
    _cutoff_panel(fig.add_subplot(gs[1, 1]), form)
    return fig, counts


if __name__ == "__main__":
    import sys
    which, scf_in, scf_out, image = sys.argv[1:5]
    fig, counts = make_sio2_figure(which, Path(scf_in).read_text(), Path(scf_out).read_text(errors="ignore"))
    print(counts["npw_k"], counts["n_rho"], counts["n_smooth"], "differences:", counts["differences"])
    fig.savefig(image, dpi=100)
