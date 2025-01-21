import os


def get_energy_oszicar(job_path):
    """
    Reads the final total energy from the OSZICAR file in job_path.
    Raises FileNotFoundError if the OSZICAR file does not exist.
    """
    oszicar_file = os.path.join(job_path, "OSZICAR")
    if not os.path.isfile(oszicar_file):
        raise FileNotFoundError(f"OSZICAR file not found in {job_path}")

    with open(oszicar_file, 'r') as infile:
        lines = infile.readlines()

    for line in reversed(lines):
        if ' F= ' in line:
            energy = float(line.split()[4])
            return energy

    return None
