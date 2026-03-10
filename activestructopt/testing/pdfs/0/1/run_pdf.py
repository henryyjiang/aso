from diffpy.structure.lattice import Lattice
from diffpy.structure.atom import Atom
from diffpy.structure.structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
import numpy as np

def main():
    lat = Lattice(a=21.995380130000004, b=7.33179338, c=7.33179338, alpha=109.47122063, beta=109.47122063, gamma=109.47122063)

    atoms = [Atom(atype = 'Na', xyz = [0.16666667, 0.0, 0.4513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.5, 0.0, 0.4513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.83333333, 0.0, 0.4513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.15044337, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.4837767, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.81711003, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.0, 0.9513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.3333333333333333, 0.9513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.6666666666666666, 0.9513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.31711003, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.65044337, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.9837767, 0.5, 0.0], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.0, 0.4513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.3333333333333333, 0.4513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.6666666666666666, 0.4513301, 0.5], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.16666667, 0.0, 0.9513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.5, 0.0, 0.9513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.83333333, 0.0, 0.9513301], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.18288997, 0.5486699, 0.5486699], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.5162233, 0.5486699, 0.5486699], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.84955663, 0.5486699, 0.5486699], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.0162233, 0.0486699, 0.0486699], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.34955663, 0.0486699, 0.0486699], occupancy = 1.0, lattice = lat),Atom(atype = 'Na', xyz = [0.68288997, 0.0486699, 0.0486699], occupancy = 1.0, lattice = lat),]

    bisoequiv = 1.0
    qmax = 18
    qmin = 1
    rmax = 10
    rmin = 1
    outfile = '/storage/home/hcoda1/7/hjiang378/aso/activestructopt/testing/pdfs/0/1/pdf_out.txt'

    total_struct = Structure(atoms = atoms, lattice = lat)
    total_struct.Bisoequiv = bisoequiv

    pdfc0 = PDFCalculator(qmax = qmax, qmin = qmin, rmax = rmax, rmin = rmin)

    _, g0_total = pdfc0(total_struct)

    ind_contribs = []

    for j in range(len(atoms)):
        missing_atom_struct = Structure(atoms = [atoms[i] for i in np.where(np.arange(len(atoms)) != j)[0]], lattice = lat)
        missing_atom_struct.Bisoequiv = bisoequiv

        _, g0_ind = pdfc0(missing_atom_struct)
        ind_contribs.append(g0_total - g0_ind)

    # correct for broadening
    new_ind_contribs = len(atoms) * (np.array(ind_contribs) + np.repeat(
        ((g0_total - np.sum(np.array(ind_contribs), axis = 0)) / len(atoms))[np.newaxis, :], len(atoms), axis = 0))
    np.savetxt(outfile, new_ind_contribs)

if __name__ == "__main__":
    main()
