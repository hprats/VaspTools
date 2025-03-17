import os
import warnings
import numpy as np

from math import pi
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from .potcar_library_path import get_potcar_library_path
from .vasp_recommended_pp import VASP_RECOMMENDED_PP

# Load valid INCAR tags from valid_incar_tags.txt file
this_dir = os.path.dirname(__file__)
valid_incar_file = os.path.join(this_dir, 'valid_incar_tags.txt')

with open(valid_incar_file, 'r') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
VALID_INCAR_TAGS = set(lines)

# Load path to the potcar library
PP_LIBRARY_PATH = get_potcar_library_path()


class StructureOptimization:
    """
    Prepare input files (INCAR, KPOINTS, POSCAR, POTCAR) for VASP structure optimization
    or other calculations (e.g., gas-phase molecules).

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        The ASE Atoms object describing the structure.
    incar_tags : dict
        Dictionary of INCAR tags, e.g. {'ISIF': 3, 'IBRION': 2, ...}.
    magmom : dict, optional
        Dictionary mapping element symbols to initial magnetic moments.
        If provided, the MAGMOM tag is added to the INCAR file.
        For elements not included, a default of 0.0 is assumed.
    kspacing : float, optional
        The smallest allowed k-point spacing in Å^-1. Default is 1.0.
        Recommended: 0.66 for bulk optimization, 1.0 for the rest.
        This should not be provided if `periodicity=None`.
    kpointstype : str, optional
        Either 'gamma' (Gamma-centered) or 'mp' (Monkhorst-Pack). Default is 'gamma'.
        This should not be provided if `periodicity=None`.
    potcar_dict : dict, optional
        Dictionary mapping element symbol -> POTCAR subfolder name.
        Defaults to VASP_RECOMMENDED_PP if not specified.
    periodicity : str or None, optional
        '2d' (slab), '3d' (bulk), or None for non-periodic (gas-phase) calculations.
        Default is '2d'.

    Examples
    --------
    >>> from ase.build import bulk
    >>> atoms = bulk('Si')
    >>> incar_tags = {'ISIF': 3, 'IBRION': 2, 'ENCUT': 520}
    >>> magmom = {'Co': 1.67}
    >>> job = StructureOptimization(atoms, incar_tags, magmom, 0.5, 'gamma', VASP_RECOMMENDED_PP, '3d')
    >>> job.write_input_files(folder_name='test_si_opt')
    """

    def __init__(
        self,
        atoms,
        incar_tags,
        magmom=None,
        kspacing=1.0,
        kpointstype='gamma',
        potcar_dict=VASP_RECOMMENDED_PP,
        periodicity='2d',
    ):
        self.atoms = atoms
        self.incar_tags = incar_tags
        self.periodicity = periodicity
        self.magmom = magmom

        if self.magmom is not None:
            ispin_value = self.incar_tags.get('ISPIN', None)
            if ispin_value != 2:
                warnings.warn(
                    f"When providing a magmom dictionary, the ISPIN tag should be set to 2. Found ISPIN={ispin_value}.",
                    UserWarning
                )

        if self.periodicity is None:
            if kspacing != 1.0 or kpointstype != 'gamma':
                raise ValueError(
                    "For periodicity=None (gas-phase), do not provide kspacing or kpointstype."
                )
            self.kspacing = None
            self.kpointstype = None
        else:
            self.kspacing = float(kspacing)
            self.kpointstype = kpointstype.lower()

        if not isinstance(potcar_dict, dict):
            raise ValueError("potcar_dict must be a dictionary.")
        self.potcar_dict = potcar_dict

        # Validate INCAR tags
        self._check_incar_tags()

        # Check periodicity + ISIF
        self._check_periodicity_incar()

    def write_input_files(self, folder_name="bulk"):
        """
        Write INCAR, KPOINTS, POSCAR, and POTCAR to `folder_name`.
        If the folder already exists, print a warning and do nothing.

        Parameters
        ----------
        folder_name : str, optional
            The directory in which to write the files. Default is 'bulk'.
        """
        if os.path.isdir(folder_name):
            warnings.warn(
                f"The folder '{folder_name}' already exists. Doing nothing.",
                UserWarning
            )
            return

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
        Write the INCAR file based on the incar_tags dictionary and include MAGMOM tag if applicable.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the INCAR.
        """
        incar_path = os.path.join(folder_name, "INCAR")
        with open(incar_path, "w") as f:
            for tag_key, tag_val in self.incar_tags.items():
                f.write(f"{tag_key} = {tag_val}\n")
            if self.magmom is not None:
                magmom_str = self._generate_magmom_string()
                if magmom_str:
                    f.write(f"MAGMOM = {magmom_str}\n")

    def _generate_magmom_string(self):
        """
        Generate the MAGMOM string based on the atoms' ordering and the magmom dictionary.
        For elements not specified in the provided dictionary, a default of 0.0 is used.
        Returns an empty string if all magmom values are zero.
        """
        symbols = self.atoms.get_chemical_symbols()
        if not symbols:
            return ""

        # Build unique element list and their counts.
        # This assumes that the atoms are grouped by element as in the POSCAR.
        unique_elements = []
        counts = []
        current_element = symbols[0]
        count = 1
        for sym in symbols[1:]:
            if sym == current_element:
                count += 1
            else:
                unique_elements.append(current_element)
                counts.append(count)
                current_element = sym
                count = 1
        unique_elements.append(current_element)
        counts.append(count)

        magmom_entries = []
        zeros_count = 0
        for elem, cnt in zip(unique_elements, counts):
            mag = self.magmom.get(elem, 0.0)
            if mag == 0.0:
                zeros_count += cnt
            else:
                if zeros_count > 0:
                    magmom_entries.append(f"{zeros_count}*0.0")
                    zeros_count = 0
                magmom_entries.append(f"{cnt}*{mag}")
        if zeros_count > 0:
            magmom_entries.append(f"{zeros_count}*0.0")
        return " ".join(magmom_entries)

    def _write_kpoints(self, folder_name):
        """
        Write the KPOINTS file. If periodicity is None (gas-phase),
        write a single k-point (Gamma only). Otherwise, compute
        the k-point mesh from user-specified kspacing and type.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the KPOINTS file.
        """
        kpoints_path = os.path.join(folder_name, "KPOINTS")

        # Non-periodic case: single k-point
        if self.periodicity is None:
            content = """Gamma-point only
 0
Monkhorst Pack
 1 1 1
 0 0 0
"""
            with open(kpoints_path, "w") as f:
                f.write(content)
            return

        # Periodic case: 2D or 3D
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

        with open(kpoints_path, "w") as f:
            f.write(kpoints_content)

    def _write_potcar(self, folder_name):
        """
        Write the POTCAR file by concatenating the appropriate POTCAR files for each element.

        Parameters
        ----------
        folder_name : str
            The directory in which to write the POTCAR.
        """
        symbols = self.atoms.get_chemical_symbols()  # e.g. ['Al', 'W', 'C', ...]
        if not symbols:
            raise ValueError("No elements found to build POTCAR.")

        # Build a minimal list of elements in the correct order (avoid duplicates in a row).
        current_element = symbols[0]
        elements_for_potcar = [current_element]
        for element in symbols[1:]:
            if element != current_element:
                elements_for_potcar.append(element)
                current_element = element

        # Concatenate POTCARs by shelling out a `cat`.
        cmd = "cat"
        for element in elements_for_potcar:
            pot_subfolder = self.potcar_dict.get(element)
            if pot_subfolder is None:
                raise ValueError(f"No POTCAR mapping found for element '{element}' in `potcar_dict`.")
            potcar_path = os.path.join(PP_LIBRARY_PATH, pot_subfolder, "POTCAR")
            if not os.path.exists(potcar_path):
                raise FileNotFoundError(f"POTCAR not found at {potcar_path}")
            cmd += f" {potcar_path}"

        potcar_out = os.path.join(folder_name, "POTCAR")
        os.system(f"{cmd} > {potcar_out}")

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
        If periodicity=None (gas-phase), we do not impose any specific ISIF.
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
        elif self.periodicity is None:
            # Gas-phase: no restrictions on ISIF.
            return
        else:
            warnings.warn(
                f"Unknown periodicity '{self.periodicity}'. Expected '2d', '3d', or None.",
                UserWarning
            )
