import os
import warnings
import numpy as np
from math import pi, ceil
from ase.io import write
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


def _is_comment_key(key) -> bool:
    """Return True if the key should be treated as a comment line in INCAR."""
    return isinstance(key, str) and key.lstrip().startswith('#')


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
    kspacing : float or None, optional
        If provided (not None), the smallest allowed k-point spacing in Å^-1.
        Mutually exclusive with `kpoints` and with providing a 'KSPACING' tag in `incar_tags`.
        If None, VASP k-point generation must be specified either via `kpoints` or via
        a 'KSPACING' entry in `incar_tags`.
    kpoints : tuple(int, int, int) or None, optional
        Explicit k-point mesh, e.g. (5, 5, 1) for a 5x5x1 grid.
        Mutually exclusive with `kspacing` and with providing a 'KSPACING' tag in `incar_tags`.
        For 2D periodicity, the third value is forced to 1.
        Only applicable if kspacing is not None. Ignored otherwise.
    kspacing_definition : {'vasp', 'pymatgen'}, optional
        How the provided kspacing value should be interpreted.
        - 'pymatgen' (default): Assumes the reciprocal lattice vectors already
          include a 2*pi factor (as returned by pymatgen).
        - 'vasp': Uses the VASP convention where the reciprocal lattice vectors do
          not have the extra 2*pi. In this case, the number of k-points along a direction
          is computed as: n_i = max(1, int(|b_i|/kspacing + 0.5)).
        Only applicable if kspacing is not None. Ignored otherwise.
    kpointstype : str, optional
        Either 'gamma' (Gamma-centered) or 'mp' (Monkhorst-Pack). Default is 'gamma'.
    potcar_dict : dict, optional
        Dictionary mapping element symbol -> POTCAR subfolder name.
        Defaults to VASP_RECOMMENDED_PP if not specified.
    periodicity : str or None, optional
        '2d' (slab), '3d' (bulk), or None for non-periodic (gas-phase) calculations.
        If periodicity is None (gas-phase), a single Gamma point is used.
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
        kspacing=None,
        kpoints=None,
        kspacing_definition='pymatgen',
        kpointstype='gamma',
        potcar_dict=VASP_RECOMMENDED_PP,
        periodicity='2d',
    ):
        self.atoms = atoms
        self.incar_tags = incar_tags
        self.periodicity = periodicity
        self.magmom = magmom

        # Set the k-point generation options.
        # Exactly one of the following three options must be used:
        #   1) `kspacing` (write a KPOINTS file based on spacing)
        #   2) `kpoints`  (write a KPOINTS file with an explicit mesh)
        #   3) 'KSPACING' in `incar_tags` (let VASP generate k-points; do not write KPOINTS)
        self.kspacing = kspacing
        self.kpoints = kpoints

                # `kspacing_definition` is only relevant when `kspacing` is used to generate the mesh.
        # If `kspacing` is None (explicit `kpoints` or INCAR `KSPACING`), this flag is ignored.
        self.kspacing_definition = kspacing_definition.lower() if kspacing_definition is not None else None
        if self.kspacing is not None:
            if self.kspacing_definition not in ['vasp', 'pymatgen']:
                raise ValueError("kspacing_definition must be either 'vasp' or 'pymatgen'.")

        if self.magmom is not None:
            ispin_value = self.incar_tags.get('ISPIN', None)
            if ispin_value != 2:
                warnings.warn(
                    f"When providing a magmom dictionary, the ISPIN tag should be set to 2. Found ISPIN={ispin_value}.",
                    UserWarning
                )

        # Validate k-point settings (mutual exclusivity between kspacing, kpoints, and INCAR KSPACING)
        has_incar_kspacing = 'KSPACING' in self.incar_tags

        if has_incar_kspacing:
            if self.kspacing is not None or self.kpoints is not None:
                raise ValueError(
                    "Found 'KSPACING' in incar_tags. In this case, both `kspacing` and `kpoints` must be None."
                )
            # No KPOINTS file will be written; VASP will generate k-points from INCAR.
            self.kpointstype = None

        else:
            # No KSPACING tag in INCAR. The user must provide either kspacing or kpoints.
            if self.kspacing is not None and self.kpoints is not None:
                raise ValueError("`kspacing` and `kpoints` are mutually exclusive. Please provide only one of them.")

            if self.kspacing is None and self.kpoints is None:
                raise ValueError(
                    "No k-point specification provided. Please provide either `kspacing`, `kpoints`, "
                    "or include 'KSPACING' in incar_tags."
                )

            # Validate and normalize kpoints if provided
            if self.kpoints is not None:
                if not (isinstance(self.kpoints, (tuple, list)) and len(self.kpoints) == 3):
                    raise ValueError("`kpoints` must be a tuple/list of three integers, e.g. (5, 5, 1).")
                try:
                    self.kpoints = tuple(int(x) for x in self.kpoints)
                except Exception as exc:
                    raise ValueError("`kpoints` must contain integers only.") from exc
                if any(k <= 0 for k in self.kpoints):
                    raise ValueError("`kpoints` must contain positive integers only.")

            # Validate kspacing if provided
            if self.kspacing is not None:
                # For periodic calculations (2d or 3d), ensure kspacing is a float.
                self.kspacing = float(self.kspacing)

            # kpointstype is only relevant when a KPOINTS file is written
            self.kpointstype = kpointstype.lower()

        # Additional periodicity-related adjustments
        if self.periodicity is None:
            # For non-periodic (gas-phase) calculations, a single Gamma point will be used when writing KPOINTS.
            # If the user provided an explicit mesh, it is ignored and replaced by (1, 1, 1).
            if self.kpoints is not None and self.kpoints != (1, 1, 1):
                warnings.warn(
                    f"periodicity=None (gas-phase). Using a single Gamma point (1, 1, 1) instead of kpoints={self.kpoints}.",
                    UserWarning
                )
                self.kpoints = (1, 1, 1)

        elif self.periodicity == '2d':
            # For 2D periodicity, force the number of k-points in the z-direction to 1.
            if self.kpoints is not None and self.kpoints[2] != 1:
                warnings.warn(
                    f"periodicity='2d'. Forcing kpoints z-component to 1 (was {self.kpoints[2]}).",
                    UserWarning
                )
                self.kpoints = (self.kpoints[0], self.kpoints[1], 1)
        if not isinstance(potcar_dict, dict):
            raise ValueError("potcar_dict must be a dictionary.")
        self.potcar_dict = potcar_dict

        # Validate INCAR tags
        self._check_incar_tags()

        # Check periodicity + ISIF
        self._check_periodicity_incar()

    def write_input_files(self, folder_name="bulk"):
        """
        Write INCAR, KPOINTS (if kspacing is provided), POSCAR, and POTCAR to `folder_name`.
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
        """
        poscar_path = os.path.join(folder_name, "POSCAR")
        write(poscar_path, self.atoms, format="vasp", direct=True, vasp5=True)

    def _write_incar(self, folder_name):
        """
        Write the INCAR file based on the incar_tags dictionary and include MAGMOM tag if applicable.
        """
        incar_path = os.path.join(folder_name, "INCAR")
        with open(incar_path, "w") as f:
            for tag_key, tag_val in self.incar_tags.items():
                if _is_comment_key(tag_key):
                    # Write comments as-is, no '=' and no value
                    f.write(f"{tag_key}\n")
                else:
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
        Write the KPOINTS file.
        """
        kpoints_path = os.path.join(folder_name, "KPOINTS")

        # Case 1: Let VASP generate k-points from INCAR (no KPOINTS file written).
        if self.kpointstype is None:
            return

        # Case 2: Explicit k-point mesh provided by the user.
        if self.kpoints is not None:
            n1, n2, n3 = self.kpoints
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
            return

        # Case 3: If periodicity is None (non-periodic: gas-phase) AND kspacing is used,
        # write a single Gamma point.
        if self.periodicity is None:
            content = """Gamma-point only
0
Gamma
1 1 1
0 0 0
"""
            with open(kpoints_path, "w") as f:
                f.write(content)
            return
        # Case 4: Compute k-point mesh from kspacing (periodic systems: 2d or 3d).
        if self.kspacing is None:
            raise ValueError("Internal error: expected `kspacing` to be set when `kpoints` is None and `kpointstype` is not None.")

        cell = np.array(self.atoms.get_cell())
        volume = np.dot(cell[0], np.cross(cell[1], cell[2]))

        # Calculate the number of k-points according to the specified definition.
        if self.kspacing_definition == 'pymatgen':
            # Using pymatgen's convention: n_i = max(1, int((2π/kspacing)*|b_i| + 0.5))

            # Compute the reciprocal lattice vectors:
            b1 = 2 * pi * np.cross(cell[1], cell[2]) / volume
            b2 = 2 * pi * np.cross(cell[2], cell[0]) / volume
            b3 = 2 * pi * np.cross(cell[0], cell[1]) / volume

            # Determine the lengths of the reciprocal lattice vectors.
            b1_len = np.linalg.norm(b1)
            b2_len = np.linalg.norm(b2)
            b3_len = np.linalg.norm(b3)

            Rk = 2 * pi / self.kspacing
            n1 = max(1, int(Rk * b1_len + 0.5))
            n2 = max(1, int(Rk * b2_len + 0.5))
            n3 = max(1, int(Rk * b3_len + 0.5))

        elif self.kspacing_definition == 'vasp':
            # Using VASP's convention:
            # N_i = max(1, ceiling(|b_i| * 2π / kspacing))
            b1 = np.cross(cell[1], cell[2]) / volume
            b2 = np.cross(cell[2], cell[0]) / volume
            b3 = np.cross(cell[0], cell[1]) / volume
            b1_len = np.linalg.norm(b1)
            b2_len = np.linalg.norm(b2)
            b3_len = np.linalg.norm(b3)
            n1 = max(1, ceil(b1_len * 2 * pi / self.kspacing))
            n2 = max(1, ceil(b2_len * 2 * pi / self.kspacing))
            n3 = max(1, ceil(b3_len * 2 * pi / self.kspacing))

        else:
            raise ValueError("kspacing_definition must be either 'vasp' or 'pymatgen'.")

        # For 2D periodicity, force the number of k-points in the z-direction to 1.
        if self.periodicity == '2d':
            n3 = 1

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
        Warn if any unrecognized (non-comment) tags are found.
        """
        user_keys = set(self.incar_tags.keys())
        # Exclude comment keys from validation
        non_comment_keys = {k for k in user_keys if not _is_comment_key(k)}
        invalid_keys = non_comment_keys - set(VALID_INCAR_TAGS)
        if invalid_keys:
            warnings.warn(
                f"The following INCAR tags are not recognized as standard VASP tags: {invalid_keys}",
                UserWarning
            )

    def _check_periodicity_incar(self):
        """
        Check that `ISIF` is consistent with the specified periodicity.
        For 3D, ISIF should be 3.
        For 2D, ISIF should be 0 or 4 (or omitted).
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
            # Allow ISIF = 0, 4 or not set; warn otherwise
            if isif_value is not None and isif_value not in (0, 4):
                warnings.warn(
                    f"Recommended ISIF=0 or 4 (or omit it) for 2D periodicity, but got ISIF={isif_value}.",
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
