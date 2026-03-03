from activestructopt.simulation.pdf import PDF
from activestructopt.active.config import pso_config
from activestructopt.active.active import ActiveLearning
import copy
import numpy as np
import sys
import os
from pymatgen.io.cif import CifParser

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = pso_config
    config['aso_params']['sampler']['name'] = 'Perturbation'
    config['aso_params']['sampler']['args'] = {'perturbrmin': 0.1, 'perturbrmax': 1.0}
    config['aso_params']['optimizer']['args'] = {
    'particles': 10,
    'iters': 10,
    'c1': 0.8,
    'c2': 0.8,
    'w': 0.4,
    'optimize_atoms': True,
    'optimize_lattice': True,
    'constraint_scale': 1.0,
    'save_obj_values': True,
    'use_torchsim': True,
   'torchsim_model_type': 'mattersim',
   'torchsim_model_path': 'mattersim-v1.0.0-5M.pth',
   'torchsim_optimizer': 'frechet_cell_fire',
   'local_steps': 25,
   }

    config['aso_params']['max_forward_calls'] = 100
    config['aso_params']['optimizer']['switch_profiles'] = [20]
    config['aso_params']['optimizer']['switch_opt_profiles'] = [20]
    config['aso_params']['optimizer']['opt_profiles'] =  [{'starts': 128, 'iters_per_start': 100,}, {'starts': 1024, 'iters_per_start': 1000,}]
    config['aso_params']['model']['profiles'] = [{'iterations': 500, 'lr': 0.001}]
    #config['aso_params']['model']['profiles'] = [{'iterations': 10, 'lr': 0.001}]
    
    test_num = "1"
    progress_dir = os.path.join(script_dir, "active_res_progress" + test_num)
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)

    pristine_structure = CifParser("starting/" + str(sys.argv[1]) + ".cif").get_structures(primitive = False)[0]
    target_structure = CifParser("target/" + str(sys.argv[1]) + ".cif").get_structures(primitive = False)[0]

    pdf_folder = os.path.join(script_dir, "pdfs", str(sys.argv[1]))
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)

    rdf_func = PDF(pristine_structure, bisoequiv = 1.0, qmax = 18, qmin = 1, rmax = 10, rmin = 1,
            python = os.path.expanduser("~/conda_envs/diffpy37/bin/python"),
            folder = pdf_folder)

    target_promise = copy.deepcopy(rdf_func)
    target_promise.get(target_structure)
    target_spec = target_promise.resolve()
    target_spec = np.mean(target_spec[np.array(rdf_func.mask)], axis = 0)

    # Check for existing progress to resume, otherwise start fresh
    progress_files = list(filter(lambda x: x.startswith(str(sys.argv[1]) + "_"), os.listdir(progress_dir)))
    if progress_files:
        filename = progress_files[0]
        if "_199" in filename:
            print(f"Structure {sys.argv[1]} already completed (iteration 199), skipping.")
            sys.exit()
        al = ActiveLearning(rdf_func, target_spec, pristine_structure, config = config, verbosity = 0, index = sys.argv[1], target_structure = target_structure,
                progress_file = os.path.join(progress_dir, filename), override_config = True)
    else:
        al = ActiveLearning(rdf_func, target_spec, pristine_structure, config = config, verbosity = 0, index = sys.argv[1], target_structure = target_structure)

    al.optimize(save_progress_dir = progress_dir)

if __name__ == "__main__":
    main()