import os


def get_energy_oszicar(job_path):
    """
    Reads the final total energy from the OSZICAR file in job_path.

    Raises:
        FileNotFoundError: if OSZICAR file does not exist.
        ValueError: if the final energy cannot be parsed from the file.
    """
    oszicar_file = os.path.join(job_path, "OSZICAR")
    if not os.path.isfile(oszicar_file):
        raise FileNotFoundError(f"OSZICAR file not found in {job_path}")

    with open(oszicar_file, 'r') as infile:
        lines = infile.readlines()

    for line in reversed(lines):
        if ' F= ' in line:
            try:
                energy = float(line.split()[4])
                return energy
            except (IndexError, ValueError):
                raise ValueError(
                    f"Failed to parse total energy from line '{line.strip()}' "
                    f"in OSZICAR at {job_path}"
                )

    raise ValueError(f"Could not find a line containing ' F= ' in OSZICAR at {job_path}")
