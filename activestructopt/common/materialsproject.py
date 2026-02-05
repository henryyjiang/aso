import requests
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
import numpy as np

MP_BASE_URL = "https://api.materialsproject.org/"

def get_structure(mpid, api_key):
    headers = {'accept': 'application/json', 'X-API-KEY': api_key}
    query = {'material_ids': mpid, '_fields': 'structure'}
    response = requests.get(MP_BASE_URL + "materials/summary", 
                             params = query, headers = headers)
    return Structure.from_dict(response.json()['data'][0]['structure'])

def get_random_structures(stoichiometry, N, api_key):
    headers = {'accept': 'application/json', 'X-API-KEY': api_key}
    structures = []
    done = False
    i = 0
    nsites = int(stoichiometry.num_atoms)
    while not done:
        query = {
        'nsites': nsites,
        '_skip': i * 1000,
        '_limit': 1000,
        '_fields': 'structure'
        }
        response = requests.get(MP_BASE_URL + "materials/summary", 
                            params = query, headers = headers)
        done = len(response.json()['data']) < 1000
        i = i + 1
        structures.extend([Structure.from_dict(
            d['structure']) for d in response.json()['data']])
    structs = [structures[i] for i in np.random.choice(
        range(len(structures)), N)]
    comp_indices = np.arange(nsites)
    np.random.shuffle(comp_indices)
    for s in structs:
        for i in range(nsites):
            j = i + 1
            for k in stoichiometry.keys():
                j -= stoichiometry[k]
                if j <= 0:
                    s.sites[comp_indices[i]].species = Composition(
                        k.symbol + '1')
                    break
    return structs
