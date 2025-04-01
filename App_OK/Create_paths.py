import os

def find_matching_file(image_name, prefix, directory,extension):
    """
    Găsește un fișier care corespunde exact cu partea de după "Right_" din numele imaginii.

    :param image_name: Numele fișierului imaginii, de exemplu, "Right_pendulum".
    :param prefix: Prefixul fișierului de căutat, de exemplu, "calib_" sau "disp0_".
    :param directory: Directorul unde se caută fișierul.
    :return: Numele fișierului găsit sau un mesaj de eroare.
    """
    print(image_name)
    if not isinstance(image_name, str) or "Right_" not in image_name:
        return f"Format invalid pentru imagine: {image_name}"
    suffix = image_name.split("Right_")[1].lower()  # Extragem și convertim în lowercase
    expected_file = f"{prefix}{suffix.split('.')[0]}{extension}"
    print(directory)
    # Listăm fișierele din director și le verificăm insensibil la majuscule
    files = {f.lower(): f for f in os.listdir(directory)}

    return files.get(expected_file, f"Nu s-a găsit fișierul {expected_file}.")

# Funcții care apelează find_matching_file cu prefixele corecte
def find_calibration_file(image_name, directory="./calibs"):
    return find_matching_file(image_name, "calib_", directory, extension=".txt")

def find_disps_file(image_name, directory="./disps"):
    return find_matching_file(image_name, "disp0_", directory, extension=".pfm")

# Funcția principală
def run(paths):
    print(paths)
    if not paths or not isinstance(paths, list):
        return "Lista de căi este invalidă."

    calib_file = find_calibration_file(paths[1])
    gt_disp_path = find_disps_file(paths[1])

    print(f"Fișierul de calibrare ales: {calib_file}")
    print(f"Fișierul disparity ales: {gt_disp_path}")

    return calib_file, gt_disp_path

# Exemplu de utilizare
#image_paths = ["Right_pendulum"]
#run(image_paths)
