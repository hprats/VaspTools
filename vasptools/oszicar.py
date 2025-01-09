def get_energy_oszicar(job_path):
    energy = None
    with open(f"{job_path}/OSZICAR") as infile:
        lines = infile.readlines()
    final = len(lines) - 1
    for i in range(final, 0, -1):
        if ' F= ' in lines[i]:
            energy = float(lines[i].split()[4])
            break
    return energy
