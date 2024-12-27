import os
import warnings
import numpy as np

from math import pi
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from .user_data import PP_PATH
from .vasp_recommended_pp import VASP_RECOMMENDED_PP

# Load valid INCAR tags from valid_incar_tags.txt
this_dir = os.path.dirname(__file__)
valid_incar_file = os.path.join(this_dir, 'valid_incar_tags.txt')

with open(valid_incar_file, 'r') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
VALID_INCAR_TAGS = set(lines)


def write_potcar(job_path, poscar_elements, pp_dict, pp_path):
    """
    Writes the POTCAR file for the elements in poscar_elements, in order.

    Parameters
    ----------
    job_path : str
        Path to the output folder.
    poscar_elements : list of str
        Chemical symbols in the order they appear in the POSCAR.
    pp_dict : dict
        Mapping from element symbol -> subfolder name. E.g. {'Al': 'Al', 'W': 'W_pv'}.
    pp_path : str
        Path to the folder containing all POTCAR subfolders (user-specific).

    Raises
    ------
    ValueError
        If `poscar_elements` is empty or no mapping is found for a given element.
    FileNotFoundError
        If the required POTCAR file does not exist.
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


class StructureOptimization:
    """
    Prepare input files (INCAR, KPOINTS, POSCAR, POTCAR) for VASP structure optimization.

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        The ASE Atoms object describing the structure.
    incar_tags : dict
        Dictionary of INCAR tags, e.g. {'ISIF': 3, 'IBRION': 2, ...}.
    kspacing : float, optional
        The smallest allowed k-point spacing in Å^-1. Default is 1.0.
        Recommended: 0.66 for bulk optimization, 1.0 for the rest.
    kpointstype : str, optional
        Either 'gamma' (Gamma-centered) or 'mp' (Monkhorst-Pack). Default is 'gamma'.
    potcar_dict : dict, optional
        Dictionary mapping element symbol -> POTCAR subfolder name. Default is VASP recommended PP.
    periodicity : str, optional
        '2d' (slab) or '3d' (bulk). Default is '2d'.

    Examples
    --------
    >>> from ase.build import bulk
    >>> atoms = bulk('Si')
    >>> incar_tags = {'ISIF': 3, 'IBRION': 2, 'ENCUT': 520}
    >>> job = StructureOptimization(atoms, incar_tags, 0.5, 'gamma', {}, '3d')
    >>> job.write_input_files(folder_name='test_si_opt')
    """

    def __init__(
        self,
        atoms,
        incar_tags,
        kspacing=1.0,
        kpointstype='gamma',
        potcar_dict=VASP_RECOMMENDED_PP,
        periodicity='2d',
    ):
        self.atoms = atoms
        self.incar_tags = incar_tags
        self.kspacing = float(kspacing)
        self.kpointstype = kpointstype.lower()
        self.periodicity = periodicity.lower()

        if potcar_dict is None:
            potcar_dict = {}
        self.potcar_dict = potcar_dict

        # Validate INCAR tags
        self._check_incar_tags()

        # Check periodicity + ISIF
        self._check_periodicity_incar()

    def write_input_files(self, folder_name="bulk"):
        """
        Write INCAR, KPOINTS, POSCAR, and POTCAR to `folder_name`.

        Parameters
        ----------
        folder_name : str, optional
            The directory in which to write the files. Default is 'bulk'.
        """
        os.makedirs(folder_name, exist_ok=True)
        self._write_poscar(folder_name)
        self._write_incar(folder_name)
        self._write_kpoints(folder_name)
        self._write_potcar(folder_name)

    def _write_poscar(self, folder_name):
        """
        Write the POSCAR file using ASE’s VASP writer.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the POSCAR.
        """
        poscar_path = os.path.join(folder_name, "POSCAR")
        write(poscar_path, self.atoms, format="vasp", direct=True, vasp5=True)

    def _write_incar(self, folder_name):
        """
        Write the INCAR file based on the incar_tags dictionary.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the INCAR.
        """
        incar_path = os.path.join(folder_name, "INCAR")
        with open(incar_path, "w") as f:
            for tag_key, tag_val in self.incar_tags.items():
                f.write(f"{tag_key} = {tag_val}\n")

    def _write_kpoints(self, folder_name):
        """
        Compute the KPOINTS mesh from the user-specified kspacing and type.

        For an orthorhombic cell with lengths a, b, c,
        Ni = max(1, int(Rk * |b_i| + 0.5)) where Rk = 2π / kspacing,
        and |b_i| are the reciprocal lattice vector lengths.

        If periodicity='2d', the number of kpoints in the z direction is fixed to 1.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the KPOINTS file.
        """
        structure = AseAtomsAdaptor().get_structure(self.atoms)
        recip = structure.lattice.reciprocal_lattice  # in Å^-1
        b1, b2, b3 = recip.matrix
        b1_len = np.linalg.norm(b1)
        b2_len = np.linalg.norm(b2)
        b3_len = np.linalg.norm(b3)

        # Rk = 2π / kspacing
        Rk = 2 * pi / self.kspacing

        n1 = max(1, int(Rk * b1_len + 0.5))
        n2 = max(1, int(Rk * b2_len + 0.5))
        n3 = max(1, int(Rk * b3_len + 0.5))

        # If 2D, fix the z-direction k-points to 1
        if self.periodicity == '2d':
            n3 = 1

        # KPOINTS style
        kpts_style = "Gamma" if self.kpointstype == 'gamma' else "Monkhorst"
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
        the correct POTCAR files.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the POTCAR.
        """
        symbols = self.atoms.get_chemical_symbols()  # e.g. ['Al', 'W', 'C', ...]
        write_potcar(
            job_path=folder_name,
            poscar_elements=symbols,
            pp_dict=self.potcar_dict,
            pp_path=PP_PATH  # from user_data.py
        )

    def _check_incar_tags(self):
        """
        Verify that all user-specified INCAR tags are recognized VASP tags.
        Warn if any unrecognized tags are found.
        """
        user_keys = set(self.incar_tags.keys())
        invalid_keys = user_keys - set(VALID_INCAR_TAGS)
        if invalid_keys:
            warnings.warn(
                f"The following INCAR tags are not recognized as standard VASP tags: {invalid_keys}",
                UserWarning
            )

    def _check_periodicity_incar(self):
        """
        Check that `ISIF` is consistent with the specified periodicity.
        For 3D, ISIF should be 3.
        For 2D, ISIF should be 0 (or omitted).
        """
        isif_value = self.incar_tags.get('ISIF', None)

        if self.periodicity == '3d':
            if isif_value != 3:
                warnings.warn(
                    f"Recommended ISIF=3 for 3D periodicity, but got ISIF={isif_value}.",
                    UserWarning
                )
        elif self.periodicity == '2d':
            if isif_value is not None and isif_value != 0:
                warnings.warn(
                    f"Recommended ISIF=0 or omit it for 2D periodicity, but got ISIF={isif_value}.",
                    UserWarning
                )
        else:
            warnings.warn(
                f"Unknown periodicity '{self.periodicity}'. Expected '2d' or '3d'.",
                UserWarning
            )
