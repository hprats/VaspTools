import os
import numpy as np

from math import pi
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from .user_data import PP_PATH


def write_potcar(job_path, poscar_elements, pp_dict, pp_path):
    """
    Writes the POTCAR file for the elements in poscar_elements, in order.
    poscar_elements: list of chemical symbols from the POSCAR (e.g. ['Al', 'W', 'C', ...])
    pp_dict: mapping from element symbol -> POTCAR subfolder (e.g. 'Al' -> 'Al', 'W' -> 'W_pv')
    pp_path: the path containing the POTCAR directories (user-specific)
    """
    if not poscar_elements:
        raise ValueError("No elements found to build POTCAR.")

    # Build a minimal list of elements in the correct order (avoid duplicates in a row).
    current_element = poscar_elements[0]
    elements_for_potcar = [current_element]
    for element in poscar_elements[1:]:
        if element != current_element:
            elements_for_potcar.append(element)
            current_element = element

    # Concatenate POTCARs by shelling out a `cat`.
    cmd = "cat"
    for element in elements_for_potcar:
        pot_subfolder = pp_dict.get(element)
        if pot_subfolder is None:
            raise ValueError(f"No POTCAR mapping found for element '{element}' in `pp_dict`.")
        potcar_path = os.path.join(pp_path, pot_subfolder, "POTCAR")
        if not os.path.exists(potcar_path):
            raise FileNotFoundError(f"POTCAR not found at {potcar_path}")
        cmd += f" {potcar_path}"

    potcar_out = os.path.join(job_path, "POTCAR")
    os.system(f"{cmd} > {potcar_out}")


class BulkOptimization:
    """
    Prepares input files for a bulk optimization in VASP.

    :param atoms: an ase.atoms object describing the structure
    :param incartags: dictionary of INCAR tags (e.g. {'ISIF': 3, 'IBRION': 2, ...})
    :param kspacing: smallest allowed k-point spacing in Å^-1
    :param kpointstype: either 'gamma' for Gamma-centered or 'mp' for Monkhorst-Pack
    :param potcar_dict: dictionary mapping element symbol -> potcar subfolder name
    """

    def __init__(
            self,
            atoms,
            incartags,
            kspacing=0.5,
            kpointstype='gamma',
            potcar_dict=None
    ):
        self.atoms = atoms
        self.incartags = incartags
        self.kspacing = float(kspacing)
        self.kpointstype = kpointstype.lower()
        if potcar_dict is None:
            potcar_dict = {}
        self.potcar_dict = potcar_dict

    def write_input_files(self, folder_name="bulk"):
        """
        Write INCAR, KPOINTS, POSCAR, and POTCAR to `folder_name`.
        """
        os.makedirs(folder_name, exist_ok=True)

        # 1) Write POSCAR
        self._write_poscar(folder_name)

        # 2) Write INCAR
        self._write_incar(folder_name)

        # 3) Write KPOINTS
        self._write_kpoints(folder_name)

        # 4) Write POTCAR
        self._write_potcar(folder_name)

    def _write_poscar(self, folder_name):
        """
        Write the POSCAR file using ASE's VASP writer.
        By default, the coordinates are written in direct (fractional) format if `direct=True`.
        """
        poscar_path = os.path.join(folder_name, "POSCAR")
        # If you prefer direct coordinates:
        write(poscar_path, self.atoms, format="vasp", direct=True, vasp5=True)

    def _write_incar(self, folder_name):
        """
        Write the INCAR file based on the incartags dictionary.
        """
        incar_path = os.path.join(folder_name, "INCAR")
        with open(incar_path, "w") as f:
            for tag_key, tag_val in self.incartags.items():
                f.write(f"{tag_key} = {tag_val}\n")

    def _write_kpoints(self, folder_name):
        """
        Compute the KPOINTS mesh from the user-specified kspacing and type.
        For an orthorhombic cell with lengths a,b,c,
        Ni = max(1, int( Rk * |b_i| + 0.5 )) where Rk = 2π / kspacing,
        and |b_i| are the reciprocal lattice vector lengths.

        For gamma-centered, we do not shift; for mp, we shift by 0.5 in each direction.
        """
        # Convert ASE atoms to a pymatgen structure so we can easily get reciprocal lattice lengths.
        structure = AseAtomsAdaptor().get_structure(self.atoms)
        recip = structure.lattice.reciprocal_lattice  # A^-1
        b1, b2, b3 = recip.matrix
        b1_len = np.linalg.norm(b1)
        b2_len = np.linalg.norm(b2)
        b3_len = np.linalg.norm(b3)

        # Rk = 2π / kspacing
        Rk = 2 * pi / self.kspacing

        # Number of subdivisions in each direction
        n1 = max(1, int(Rk * b1_len + 0.5))
        n2 = max(1, int(Rk * b2_len + 0.5))
        n3 = max(1, int(Rk * b3_len + 0.5))

        # Prepare lines for KPOINTS
        kpts_style = "Gamma" if self.kpointstype == 'gamma' else "Monkhorst"
        # For a Gamma-centered mesh, shift = 0 0 0
        # For a Monkhorst-Pack mesh, shift = 0.5 0.5 0.5
        shift_line = "0 0 0" if self.kpointstype == 'gamma' else "0.5 0.5 0.5"

        kpoints_content = f"""KPOINTS
        0
        {kpts_style}
        {n1} {n2} {n3}
        {shift_line}
        """

        kpoints_path = os.path.join(folder_name, "KPOINTS")
        with open(kpoints_path, "w") as f:
            f.write(kpoints_content)

    def _write_potcar(self, folder_name):
        """
        Gather the list of unique elements from the ASE atoms object, then concatenate
        the correct POTCAR files.  Uses the user_data.py PP_PATH by default.
        """
        symbols = self.atoms.get_chemical_symbols()  # e.g. ['Al', 'W', 'C', ...]
        write_potcar(
            job_path=folder_name,
            poscar_elements=symbols,
            pp_dict=self.potcar_dict,
            pp_path=PP_PATH  # from user_data.py
        )
