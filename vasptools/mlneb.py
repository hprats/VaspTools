"""
Module: mlneb
Description:
    Prepare input files for an ML-NEB calculation using catlearn.
    This module provides the MLNEB class, which creates a job directory
    with the following structure:
      - <job_path>/
          run.py              # Python script to run the job on the cluster.
          optimized_structures/
              initial.traj    # Optimized initial state (ASE trajectory).
              final.traj      # Optimized final state (ASE trajectory).
              list_images.traj  # NEB interpolated images (ASE trajectory).

Usage example:
    from ase.io import read
    from vasptools.mlneb import MLNEB

    elementary_step = 'CO2dis_on_Ni111'
    main_path = '/Users/hprats/PycharmProjects/project_1'

    incar_tags_ML_neb = {
        'xc': "'pbe'",
        'gga': "'PE'",
        'ivdw': 11,
        'ediff': '1e-05',
        'ediffg': -0.01,
        'nelm': 300,
        'ismear': 1,
        'sigma': 0.2,
        'lwave': 'False',
        'lcharg': 'False',
        'encut': 415,
        'algo': "'Fast'",
        'lreal': "'Auto'",
        'ldipol': 'True',
        'idipol': 3,
        'dipol': '[0.5, 0.5, 0.5]',
        'lasph': 'True',
        'ispin': 1,
    }

    initial = read(f"{main_path}/ts/{elementary_step}/initial/CONTCAR")
    final = read(f"{main_path}/ts/{elementary_step}/final/CONTCAR")

    magmom_dict = {'Co': 1.67, 'Pt': 0.0, 'C': 0.0, 'O': 0.0}
    potcar_dict = {'H': 'H', 'Ca': 'Ca_sv'}  # Example custom pseudopotential mapping

    # For periodic systems, specify kspacing and kpointstype
    job = MLNEB(
        atoms_initial=initial,
        atoms_final=final,
        incar_tags=incar_tags_ML_neb,
        n_images=10,
        fmax=0.01,
        kspacing=1.0,
        kspacing_definition='pymatgen',
        kpointstype='gamma',
        magmom=magmom_dict,
        potcar_dict=potcar_dict
    )
    job.create_job_dir(job_path=f"{main_path}/ts/{elementary_step}/mlneb")

"""

import os
import ast
from math import pi, ceil
import numpy as np
from ase.io import write
from ase.mep import NEB

class MLNEB:
    def __init__(
            self,
            atoms_initial,
            atoms_final,
            incar_tags,
            n_images,
            fmax,
            kspacing=None,
            kpointstype=None,
            kspacing_definition="pymatgen",
            magmom=None,
            potcar_dict=None
    ):
        """
        Parameters
        ----------
        atoms_initial : ase.atoms.Atoms
            The initial state structure.
        atoms_final : ase.atoms.Atoms
            The final state structure.
        incar_tags : dict
            Dictionary of INCAR tags (e.g. {'xc': "'pbe'", 'ivdw': 11, ...}).
            These will be converted into lower-case keys for the Vasp calculator.
        n_images : int
            Total number of images in the NEB interpolation (including initial and final).
        fmax : float
            fmax for the ML-NEB run (applied to the NEB optimization).
        kspacing : float or None, optional
            The desired k-point spacing in Å^-1.
            If set to None (default), no k-point tuple is generated and the run.py file
            will not include the 'kpts' and 'gamma' tags.
        kpointstype : str or None, optional
            Either 'gamma' or 'mp'; used to set the gamma flag.
            Ignored if kspacing is None.
        kspacing_definition : {'pymatgen', 'vasp'}, optional
            How the provided kspacing value should be interpreted.
            - 'pymatgen' (default): Assumes the reciprocal lattice vectors include the 2π factor.
            - 'vasp': Uses the VASP convention where the reciprocal lattice vectors do not have the extra 2π.
        magmom : dict, optional
            Dictionary mapping element symbols to initial magnetic moments.
            For elements not included, a default of 0.0 is assumed.
            The resulting MAGMOM tag is added to the INCAR file if at least one value is non-zero.
        potcar_dict : dict, optional
            Dictionary mapping element symbol -> POTCAR subfolder name.
            Defaults to {'base': 'recommended'} if not specified.
        """
        self.atoms_initial = atoms_initial
        self.atoms_final = atoms_final
        self.incar_tags = incar_tags
        self.n_images = n_images
        self.fmax = fmax
        self.magmom = magmom
        self.potcar_dict = potcar_dict if potcar_dict is not None else {'base': 'recommended'}

        if kspacing is not None:
            try:
                self.kspacing = float(kspacing)
            except Exception as e:
                raise ValueError("kspacing must be convertible to float.") from e
            if kpointstype is None:
                raise ValueError("kpointstype must be provided when kspacing is specified.")
            self.kpointstype = kpointstype.lower()
            self.kspacing_definition = kspacing_definition.lower()
            if self.kspacing_definition not in ["pymatgen", "vasp"]:
                raise ValueError("kspacing_definition must be either 'pymatgen' or 'vasp'.")
        else:
            self.kspacing = None
            self.kpointstype = None
            self.kspacing_definition = None

    def create_job_dir(self, job_path):
        """
        Creates the job directory for the ML-NEB calculation, including:
          1) The job_path folder.
          2) An 'optimized_structures' subfolder containing:
             - initial.traj and final.traj (the starting structures).
             - list_images.traj: the trajectory with initial, final, and NEB-interpolated images.
          3) A run.py file in the job_path folder that sets up the calculation.

        Parameters
        ----------
        job_path : str
            Path to the job directory to create.
        """
        # Create job folder and subfolder for optimized structures
        os.makedirs(job_path, exist_ok=True)
        opt_struct_dir = os.path.join(job_path, "optimized_structures")
        os.makedirs(opt_struct_dir, exist_ok=True)

        # Write initial and final structures as trajectory files
        initial_traj_path = os.path.join(opt_struct_dir, "initial.traj")
        final_traj_path = os.path.join(opt_struct_dir, "final.traj")
        write(initial_traj_path, self.atoms_initial, format="traj")
        write(final_traj_path, self.atoms_final, format="traj")

        # Create list_images using idpp interpolation from ASE's NEB module
        list_images = [self.atoms_initial]
        constraints = self.atoms_initial.constraints
        for _ in range(self.n_images - 2):
            image = self.atoms_initial.copy()
            image.set_constraint(constraints)
            list_images.append(image)
        list_images.append(self.atoms_final)
        neb = NEB(list_images, climb=False, k=0.5)
        neb.interpolate('idpp')
        list_images_traj_path = os.path.join(opt_struct_dir, "list_images.traj")
        write(list_images_traj_path, list_images, format="traj")

        # Process incar_tags: convert keys to lowercase and evaluate string literals if possible
        processed_tags = {}
        for key, val in self.incar_tags.items():
            key_lower = key.lower()
            if isinstance(val, str):
                try:
                    processed_val = ast.literal_eval(val)
                except Exception:
                    processed_val = val
            else:
                processed_val = val
            processed_tags[key_lower] = processed_val

        # Determine the fmax for initial and final optimizations from 'ediffg'
        initial_fmax = abs(processed_tags.get('ediffg', 0.01))

        # Build calculator parameters from processed_tags.
        # Exclude 'ediffg' because it is used only for endpoint optimization.
        calc_params_lines = []
        for key, value in processed_tags.items():
            if key == 'ediffg':
                continue
            calc_params_lines.append(f"    {key}={repr(value)},")

        # Conditionally add k-point parameters if kspacing is provided.
        if self.kspacing is not None:
            # Obtain the cell directly from the atoms (no pymatgen)
            cell = np.array(self.atoms_initial.get_cell())
            volume = np.dot(cell[0], np.cross(cell[1], cell[2]))

            if self.kspacing_definition == 'pymatgen':
                b1 = 2 * pi * np.cross(cell[1], cell[2]) / volume
                b2 = 2 * pi * np.cross(cell[2], cell[0]) / volume
                b1_len = np.linalg.norm(b1)
                b2_len = np.linalg.norm(b2)
                Rk = 2 * pi / self.kspacing
                n1 = max(1, int(Rk * b1_len + 0.5))
                n2 = max(1, int(Rk * b2_len + 0.5))

            elif self.kspacing_definition == 'vasp':
                b1 = np.cross(cell[1], cell[2]) / volume
                b2 = np.cross(cell[2], cell[0]) / volume
                b1_len = np.linalg.norm(b1)
                b2_len = np.linalg.norm(b2)
                n1 = max(1, ceil(b1_len * 2 * pi / self.kspacing))
                n2 = max(1, ceil(b2_len * 2 * pi / self.kspacing))

            else:
                raise ValueError("kspacing_definition must be 'vasp' or 'pymatgen'.")

            # For ML-NEB we assume two-dimensional periodicity so we fix n3 = 1.
            n3 = 1
            kpts_tuple = (n1, n2, n3)
            gamma_flag = True if self.kpointstype == 'gamma' else False
            calc_params_lines.append(f"    kpts={kpts_tuple},")
            calc_params_lines.append(f"    gamma={gamma_flag},")

        if self.magmom is not None:
            calc_params_lines.append("    magmom=magmom_list,")
        calc_params_str = "\n".join(calc_params_lines)

        # Prepare additional run.py lines for magmom if specified.
        magmom_lines = ""
        if self.magmom is not None:
            magmom_lines = (
                "atoms = read('./optimized_structures/initial.traj')\n"
                f"magmom_dict = {repr(self.magmom)}\n"
                "magmom_list = [magmom_dict.get(atom.symbol, 0.0) for atom in atoms]\n"
            )

        # Prepare potcar_dict lines to be added to run.py
        potcar_lines = f"potcar_dict = {repr(self.potcar_dict)}\n\n"

        # Create the run.py file content with time logging, endpoint optimizations,
        # and the ML-NEB run using catlearn.
        run_py_content = f'''from ase.io import read
from ase.optimize import BFGS
from ase.calculators.vasp import Vasp
import shutil
import copy
from catlearn.optimize.mlneb import MLNEB
from datetime import datetime

{potcar_lines}now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
with open('time.txt', 'w') as outfile:
    outfile.write(f'{{dt_string}}: starting job ...\\n')

{magmom_lines}ase_calculator = Vasp(
    setups=potcar_dict,
{calc_params_str}
)

# Optimize initial state:
now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
with open('time.txt', 'a') as outfile:
    outfile.write(f'{{dt_string}}: starting optimization of initial state ...\\n')
slab = read('./optimized_structures/initial.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='initial.traj')
qn.run(fmax={initial_fmax})
shutil.copy('./initial.traj', './optimized_structures/initial.traj')

# Optimize final state:
now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
with open('time.txt', 'a') as outfile:
    outfile.write(f'{{dt_string}}: starting optimization of final state ...\\n')
slab = read('./optimized_structures/final.traj')
slab.set_calculator(copy.deepcopy(ase_calculator))
qn = BFGS(slab, trajectory='final.traj')
qn.run(fmax={initial_fmax})
shutil.copy('./final.traj', './optimized_structures/final.traj')

# Run ML-NEB:
now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
with open('time.txt', 'a') as outfile:
    outfile.write(f'{{dt_string}}: starting ML-NEB ...\\n')
neb_catlearn = MLNEB(start='initial.traj', end='final.traj',
    ase_calc=copy.deepcopy(ase_calculator),
    n_images={self.n_images},
    interpolation='idpp',
    )
neb_catlearn.run(fmax={self.fmax}, trajectory='ML-NEB.traj')

# Print results:
now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
with open('time.txt', 'a') as outfile:
    outfile.write(f'{{dt_string}}: ML-NEB finished\\n')

from catlearn.optimize.tools import plotneb
plotneb(trajectory='ML-NEB.traj', view_path=True)
'''
        # Write the run.py file in the job_path folder
        run_py_path = os.path.join(job_path, "run.py")
        with open(run_py_path, 'w') as f:
            f.write(run_py_content)
