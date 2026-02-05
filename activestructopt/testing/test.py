from activestructopt.simulation.rdf import RDF
from activestructopt.active.config import torchmd_config, pso_config
from activestructopt.active.active import ActiveLearning
import copy
import numpy as np
import os
import sys
from pymatgen.io.cif import CifParser

def main():
	config = pso_config
	script_dir = os.path.dirname(os.path.abspath(__file__))
    
	pristine_structure = CifParser(os.path.join(script_dir, "datasets/ht/start", sys.argv[1] + ".cif")
        ).get_structures(primitive = False)[0]
	target_structure = CifParser(os.path.join(script_dir, "datasets/ht/target", sys.argv[1] + ".cif")
        ).get_structures(primitive = False)[-1]

	rdf_func = RDF(pristine_structure, σ = 0.1, max_r = 12.)

	target_promise = copy.deepcopy(rdf_func)
	target_promise.get(target_structure)
	target_spec = target_promise.resolve()
	target_spec = np.mean(target_spec[np.array(rdf_func.mask)], axis = 0)

	al = ActiveLearning(
    simfunc = rdf_func,
    target = target_spec,
    initial_structure = pristine_structure,
    index = sys.argv[1],
    config = config,
    target_structure = target_structure
)
	al.optimize(save_progress_dir = os.path.join(script_dir, 'progress'))
	al.save(os.path.join(script_dir, "res", sys.argv[1] + ".pkl"), 
		additional_data = {'target_structure': target_structure})

if __name__ == "__main__":
    main()
