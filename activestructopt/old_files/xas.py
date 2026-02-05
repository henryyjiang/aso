from pymatgen.io import feff
import os

def get_feff_inp(struct, atoms_r = 5.0, scf_r = 4.0, fms_r = 4.0, kpts = 100):
    # guarantees at least two atoms of the absorber, 
    # which is necessary because two different ipots are created
    struct.make_supercell(2)
    feff.inputs.Atoms(struct, 
        [x.symbol for x in struct.species].index('Co'), 
        atoms_r
        ).write_file()
    feff.inputs.Potential(struct, 
        [x.symbol for x in struct.species].index('Co')).write_file()
    feff.inputs.Tags({
        "TITLE": "Lithium Cobalt Oxide",
        "CONTROL": "1 1 1 1 1 1",
        "EXCHANGE": "0 0 0 0",
        "XANES": "6 0.05 0.3",
        "ABSOLUTE": "",
        "EDGE": "K -1",
        "SCF": "{} 0".format(scf_r),
        "FMS": "{} 0".format(fms_r),
        "KMESH": "{}".format(kpts),
    }).write_file()
    # https://www.geeksforgeeks.org/python-program-to-merge-two-files-into-a-third-file/
    atoms = pot = tags = ""
    with open('ATOMS') as fp:
        atoms = fp.read()
    with open('POTENTIALS') as fp:
        pot = fp.read()
    with open('PARAMETERS') as fp:
        tags = fp.read()
    with open ('feff.inp', 'w') as fp:
        fp.write(tags + '\n' + pot + '\n' + atoms)
    os.remove('ATOMS')
    os.remove('POTENTIALS')
    os.remove('PARAMETERS')
