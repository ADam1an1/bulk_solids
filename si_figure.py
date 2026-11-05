"""Explanatory figure for the silicon SCF tutorial input (pw.scf.silicon.in).

Panels
  A  Diamond-Si structure built exactly as QE builds it from ibrav=2, celldm(1), ATOMIC_POSITIONS
  B  Each hard-coded input constant -> what it means geometrically / physically
  C  Reciprocal space: the G-vectors allowed by the lattice and the two cutoff spheres
  D  Why a cutoff works: a periodic function rebuilt from plane waves up to increasing cutoff

Numbers in the panels (plane-wave counts, volume, ...) are computed here from the same
constants as the input, and checked against pw.x output when an output file is passed.
"""
from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

BOHR_A = 0.529177210903          # Angstrom per bohr
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
SEQ = ["#a9c8f0", "#5f9ae3", "#1d5fb4"]              # one-hue sequential ramp (low -> high cutoff)

# QE ibrav=2 primitive vectors (units of a) and reciprocal vectors (units of 2*pi/a), as printed by pw.x
A_PRIM = 0.5 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], float)
B_REC = np.array([[-1, -1, 1], [1, 1, 1], [-1, 1, -1]], float)


def irreducible_kpoints(mesh=(6, 6, 6)):
    """Symmetry-reduced k-points of an unshifted Monkhorst-Pack mesh (Cartesian, 2pi/a units) via spglib,
    i.e. what pw.x does with K_POINTS automatic 6 6 6 0 0 0."""
    import spglib
    cell = (A_PRIM, [[0, 0, 0], [0.25, 0.25, 0.25]] @ np.linalg.inv(A_PRIM), [14, 14])
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
    frac = grid[np.unique(mapping)] / np.array(mesh)
    return frac @ B_REC


def _kpoints(path):
    """k-points from a pw.x output if one exists (and is consistent), else from spglib."""
    kpts = irreducible_kpoints()
    if path and Path(path).exists():
        txt = Path(path).read_text()
        n = int(re.search(r"number of k points=\s+(\d+)", txt).group(1))
        if n != len(kpts):
            print(f"warning: spglib gives {len(kpts)} irreducible k-points, pw.x used {n}")
        ks = re.findall(r"k\(\s*\d+\) = \(\s*(\S+)\s+(\S+)\s+(\S+)\), wk", txt)[:n]
        return np.array(ks, float), txt
    return kpts, ""


def _g_vectors(nmax=22):
    n = np.array(list(product(range(-nmax, nmax + 1), repeat=3)), float)
    return n @ B_REC                                   # (N,3) in 2pi/a units


def plane_wave_counts(a_bohr=10.26, ecutwfc=30.0, ecutrho=None, kpts=None):
    ecutrho = ecutrho or 4 * ecutwfc
    tpiba2 = (2 * np.pi / a_bohr) ** 2                 # Ry per (2pi/a)^2, since E[Ry] = |q|^2 [bohr^-2]
    G = _g_vectors()
    kpts = np.zeros((1, 3)) if kpts is None else kpts
    npw_k = [int((((G + k) ** 2).sum(1) * tpiba2 <= ecutwfc).sum()) for k in kpts]
    n_rho = int(((G ** 2).sum(1) * tpiba2 <= ecutrho).sum())
    return dict(tpiba2=tpiba2, npw_k=npw_k, n_rho=n_rho, ecutrho=ecutrho)


def _structure_panel(ax, a_bohr):
    # Conventional cube spanning x in [-1,0], y,z in [0,1] (units of a) so QE's a1,a2,a3 lie on its faces
    fcc = [np.array(p) / 2 for p in product(range(-2, 1), range(0, 3), range(0, 3)) if sum(p) % 2 == 0]
    shift = np.array([0.25, 0.25, 0.25])
    inside = lambda p: -1 - 1e-9 <= p[0] <= 1e-9 and all(-1e-9 <= c <= 1 + 1e-9 for c in p[1:])
    subA = np.array(fcc)
    cand = [np.array(p) / 2 + shift for p in product(range(-4, 3), repeat=3) if sum(p) % 2 == 0]
    subB = np.array([p for p in cand if inside(p)])
    si2 = shift.copy()                                  # basis atom 2 itself (just outside this cube)

    # cube edges
    for s, e in [((-1, 0, 0), (0, 0, 0)), ((-1, 1, 0), (0, 1, 0)), ((-1, 0, 1), (0, 0, 1)), ((-1, 1, 1), (0, 1, 1)),
                 ((-1, 0, 0), (-1, 1, 0)), ((0, 0, 0), (0, 1, 0)), ((-1, 0, 1), (-1, 1, 1)), ((0, 0, 1), (0, 1, 1)),
                 ((-1, 0, 0), (-1, 0, 1)), ((0, 0, 0), (0, 0, 1)), ((-1, 1, 0), (-1, 1, 1)), ((0, 1, 0), (0, 1, 1))]:
        ax.plot(*zip(s, e), color=INK2, lw=0.8, alpha=0.6)

    # bonds (nearest neighbours at sqrt(3)/4 a)
    dnn = np.sqrt(3) / 4
    for b in np.vstack([subB, si2]):
        for a in subA:
            if abs(np.linalg.norm(b - a) - dnn) < 1e-6:
                ax.plot(*zip(a, b), color="#9a9890", lw=2.2, solid_capstyle="round", zorder=1)

    ax.scatter(*subA.T, s=90, color=BLUE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=3)
    ax.scatter(*subB.T, s=90, color=ORANGE, edgecolor="white", linewidth=1.5, depthshade=False, zorder=3)
    # the two basis atoms of ATOMIC_POSITIONS, ringed
    ax.scatter(*np.zeros(3), s=240, facecolor=BLUE, edgecolor=INK, linewidth=1.6, depthshade=False, zorder=4)
    ax.scatter(*si2, s=240, facecolor=ORANGE, edgecolor=INK, linewidth=1.6, depthshade=False, zorder=4)
    ax.text(0.03, -0.12, -0.10, "Si$_1$ (0, 0, 0)", color=INK, fontsize=9, weight="bold")
    ax.text(0.22, 0.30, 0.42, "Si$_2$ (¼, ¼, ¼)·a", color=INK, fontsize=9, weight="bold")

    # QE's primitive vectors
    for i, v in enumerate(A_PRIM):
        ax.quiver(0, 0, 0, *v, color=INK, lw=1.8, arrow_length_ratio=0.12, zorder=5)
        ax.text(*(v * 1.12 + np.array([0, 0, 0.06 if i == 2 else 0])), f"a$_{i + 1}$", color=INK, fontsize=10,
                weight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))

    ax.text(-0.55, -0.02, -0.09, "a", color=INK, fontsize=11, weight="bold")
    ax.text2D(0.0, -0.08, f"cube edge a = celldm(1) = {a_bohr:g} bohr = {a_bohr * BOHR_A:.3f} Å\n"
              "cube drawn over x ∈ [−a, 0], y, z ∈ [0, a] so a₁, a₂, a₃ end on face centres",
              transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    ax.set_box_aspect((1.3, 1.3, 1.6), zoom=1.15)      # proportional to the axis ranges -> undistorted cube
    ax.set_xlim(-1, 0.3); ax.set_ylim(-0.15, 1.15); ax.set_zlim(-0.15, 1.45)
    ax.view_init(elev=17, azim=-62)
    ax.set_axis_off()
    leg = [Line2D([], [], ls="", marker="o", ms=9, mfc=BLUE, mec="white", label="FCC sublattice 1 (copies of Si$_1$)"),
           Line2D([], [], ls="", marker="o", ms=9, mfc=ORANGE, mec="white",
                  label="FCC sublattice 2 = sublattice 1 + (¼,¼,¼)·a"),
           Line2D([], [], color=INK, lw=1.8, label="primitive vectors from ibrav = 2"),
           Line2D([], [], color="#9a9890", lw=2.2, label=f"Si–Si bond  √3/4·a = {dnn * a_bohr * BOHR_A:.3f} Å")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=8, frameon=False)
    ax.set_title("A   Diamond silicon as built from the input", loc="left", fontsize=11, weight="bold", color=INK)


def _mapping_panel(ax, a_bohr, ecutwfc, counts, vol):
    ax.set_axis_off()
    ax.set_title("B   What each hard-coded constant sets", loc="left", fontsize=11, weight="bold", color=INK)
    rows = [
        ("ibrav = 2", "FCC Bravais lattice. QE generates",
         "a₁ = a/2(−1,0,1)  a₂ = a/2(0,1,1)  a₃ = a/2(−1,1,0)"),
        ("celldm(1) = 10.26", f"a, the cubic edge: {a_bohr:g} bohr = {a_bohr * BOHR_A:.2f} Å (≈ experiment)",
         f"primitive-cell volume a³/4 = {vol:.1f} bohr³"),
        ("nat = 2", "atoms per primitive cell: one per FCC sublattice", "(the conventional cube would hold 8)"),
        ("ntyp = 1", "one species, Si 28.086 u, Si.pz-vbc.UPF", "LDA (SLA PZ), 4 valence e⁻ per atom"),
        ("ATOMIC_POSITIONS (alat)", "Si₁ (0, 0, 0)    Si₂ (0.25, 0.25, 0.25) × a", "Si₂ = Si₁ shifted ¼ of the cube diagonal"),
        ("nbnd = 8", "2 atoms × 4 e⁻ = 8 e⁻ → 4 filled bands", "+4 empty bands, so the gap can be read off"),
        ("K_POINTS 6 6 6 0 0 0", "6×6×6 = 216 k-points in the Brillouin zone", f"→ {len(counts['npw_k'])} after symmetry"),
        ("ecutwfc = 30 Ry", "ψ basis: plane waves with |k+G|² ≤ 30 Ry",
         f"→ {min(counts['npw_k'])}–{max(counts['npw_k'])} plane waves per k-point"),
        ("ecutrho (default)", f"4 × ecutwfc = {counts['ecutrho']:g} Ry, for ρ = Σ|ψ|²",
         f"→ {counts['n_rho']} G-vectors, FFT grid 32×32×32"),
    ]
    y = 0.95
    for key, l1, l2 in rows:
        ax.text(0.0, y, key, family="monospace", fontsize=9, weight="bold", color=INK, va="top", transform=ax.transAxes)
        ax.text(0.40, y, l1, fontsize=8.8, color=INK, va="top", transform=ax.transAxes)
        ax.text(0.40, y - 0.045, l2, fontsize=8.3, color=INK2, va="top", transform=ax.transAxes)
        ax.plot([0, 1], [y - 0.088, y - 0.088], color=GRID, lw=0.8, transform=ax.transAxes)
        y -= 0.106


def _reciprocal_panel(ax, a_bohr, ecutwfc, counts):
    tpiba2 = counts["tpiba2"]
    r_wf, r_rho = np.sqrt(ecutwfc / tpiba2), np.sqrt(counts["ecutrho"] / tpiba2)
    G = _g_vectors()
    plane = G[np.isclose(G[:, 0], G[:, 1])]            # (1-10) plane through Gamma: contains [001], [110], [111]
    u, v = plane[:, 0] * np.sqrt(2), plane[:, 2]
    g2 = (plane ** 2).sum(1)
    lim = r_rho * 1.12
    keep = (abs(u) <= lim) & (abs(v) <= lim)
    in_wf, in_rho = g2 <= r_wf ** 2, (g2 > r_wf ** 2) & (g2 <= r_rho ** 2)
    out = ~(in_wf | in_rho)
    ax.scatter(u[keep & out], v[keep & out], s=7, color="#c9c7bf", lw=0)
    ax.scatter(u[keep & in_rho], v[keep & in_rho], s=16, color=AQUA, lw=0)
    ax.scatter(u[keep & in_wf], v[keep & in_wf], s=22, color=BLUE, lw=0)
    ax.add_patch(Circle((0, 0), r_rho, fill=False, color=AQUA, lw=2))
    ax.add_patch(Circle((0, 0), r_wf, fill=False, color=BLUE, lw=2))
    ax.annotate(f"|k+G|² ≤ ecutwfc = {ecutwfc:g} Ry\n→ ψ uses {min(counts['npw_k'])}–{max(counts['npw_k'])} plane waves",
                (r_wf * 0.707, r_wf * 0.707), xytext=(r_wf * 0.35, r_rho * 0.62), fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=1),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    ax.annotate(f"|G|² ≤ ecutrho = 4·ecutwfc = {counts['ecutrho']:g} Ry\n→ ρ uses {counts['n_rho']} G-vectors",
                (-r_rho * 0.707, r_rho * 0.707), xytext=(-lim * 0.97, lim * 0.97), fontsize=8.5, color=INK,
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=AQUA, lw=1),
                arrowprops=dict(arrowstyle="-", color=AQUA, lw=1))
    ax.text(lim * 0.97, -lim * 0.97, "grey: excluded (above both cutoffs)", fontsize=8, color=INK2, ha="right",
            va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.scatter([0], [0], s=40, color=INK, zorder=5)
    ax.text(0.5, -0.5, "Γ", fontsize=10, color=INK, ha="left", va="top", weight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("G along [110]   (units of 2π/a)", fontsize=8.5, color=INK2)
    ax.set_ylabel("G along [001]   (units of 2π/a)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("C   Crystal → plane-wave basis (reciprocal lattice, slice through Γ)", loc="left",
                 fontsize=11, weight="bold", color=INK, pad=44)
    ax.text(0.0, 1.0, r"$\psi_{n\mathbf{k}}(\mathbf{r})=\sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,"
                      r"e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$,   $\mathbf{G}=n_1\mathbf{b}_1+n_2\mathbf{b}_2+n_3\mathbf{b}_3$"
                      "\nb₁ = 2π/a(−1,−1,1)  b₂ = 2π/a(1,1,1)  b₃ = 2π/a(−1,1,−1)  (BCC; set by ibrav and celldm)",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")


def _cutoff_panel(ax, a_bohr):
    # 1-D illustration along the [111] line: period sqrt(3)*a, atoms at 0 and 1/4 of it
    L = np.sqrt(3) * a_bohr
    x = np.linspace(0, L, 1200)
    sig = 0.55                                          # bohr: a sharp, atom-centred feature
    centers = np.array([0.0, 0.25]) * L
    m = np.arange(-200, 201)
    g = 2 * np.pi * m / L                                # bohr^-1 ; E = g^2 in Ry
    coef = sum(np.exp(-0.5 * (sig * g) ** 2) * np.exp(-1j * g * c) for c in centers) * sig * np.sqrt(2 * np.pi) / L
    f = lambda emax: np.real(np.exp(1j * np.outer(x, g[g ** 2 <= emax])) @ coef[g ** 2 <= emax])
    exact = f(1e9)
    for emax, c in zip([3, 10, 30], SEQ):
        n = int((g ** 2 <= emax).sum())
        ax.plot(x / L, f(emax), color=c, lw=2, label=f"cutoff {emax} Ry  ({n} plane waves)")
    ax.plot(x / L, exact, color=INK, lw=1.4, ls=(0, (4, 3)), label="target function")
    for c, lab, col in [(0, "Si$_1$", BLUE), (0.25, "Si$_2$", ORANGE), (1, "Si$_1$", BLUE)]:
        ax.scatter([c], [-0.24], s=70, color=col, edgecolor="white", lw=1.5, zorder=5)
        ax.text(c + (0.015 if c < 1 else -0.015), -0.24, lab, ha="left" if c < 1 else "right", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.32, 1.12)
    ax.set_xlabel("position along the [111] body diagonal   (units of √3·a)", fontsize=8.5, color=INK2)
    ax.set_ylabel("amplitude (arb.)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.62, 1.0))
    ax.set_title("D   What the cutoff does (1-D sketch along the Si–Si bond line)", loc="left",
                 fontsize=11, weight="bold", color=INK, pad=44)
    ax.text(0.0, 1.0, "Sharper features near the atoms need shorter-wavelength (higher-energy) plane waves.\n"
                      "Kinetic energy of a plane wave, in Ry units: E = |k+G|²; keep only E ≤ ecutwfc.",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")


def make_si_figure(pw_output=None, a_bohr=10.26, ecutwfc=30.0):
    kpts, txt = _kpoints(pw_output)
    counts = plane_wave_counts(a_bohr, ecutwfc, kpts=kpts)
    vol = abs(np.linalg.det(A_PRIM)) * a_bohr ** 3
    if txt:  # cross-check against pw.x
        m = re.search(r"Dense\s+grid:\s+(\d+) G-vectors", txt)
        if m and int(m.group(1)) != counts["n_rho"]:
            print(f"warning: density G-vectors {counts['n_rho']} != pw.x {m.group(1)}")
        m = re.search(r"unit-cell volume\s+=\s+([\d.]+)", txt)
        if m and abs(float(m.group(1)) - vol) > 1e-2:
            print(f"warning: cell volume {vol:.4f} != pw.x {m.group(1)}")

    fig = plt.figure(figsize=(15, 12.5), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1], hspace=0.32, wspace=0.12,
                          left=0.04, right=0.98, top=0.95, bottom=0.06)
    _structure_panel(fig.add_subplot(gs[0, 0], projection="3d"), a_bohr)
    _mapping_panel(fig.add_subplot(gs[0, 1]), a_bohr, ecutwfc, counts, vol)
    _reciprocal_panel(fig.add_subplot(gs[1, 0]), a_bohr, ecutwfc, counts)
    _cutoff_panel(fig.add_subplot(gs[1, 1]), a_bohr)
    return fig, counts


if __name__ == "__main__":
    import sys
    fig, counts = make_si_figure(sys.argv[1] if len(sys.argv) > 1 else None)
    print(counts["npw_k"], counts["n_rho"])
    fig.savefig(sys.argv[2] if len(sys.argv) > 2 else "si_figure.png", dpi=100)
