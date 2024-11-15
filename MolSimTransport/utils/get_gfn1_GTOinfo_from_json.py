import json

def load_orbital_data(json_file):
    """Load atomic orbital data from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def get_orbital_info(data, atomic_number):
    """
    Retrieve orbital information for a specific element using atomic number.

    Parameters:
    - data: the dictionary containing all atomic orbital data.
    - atomic_number: string, the atomic number of the element (e.g., "1" for Hydrogen, "6" for Carbon).

    Returns:
    - The orbital information string for the specified element.
    """
    try:
        return data[atomic_number]
    except KeyError:
        print(f"Orbital information for atomic number {atomic_number} not found.")
        return None

# This function is intended to be used in another script
def get_basis_set_by_atomic_number(json_file, atomic_number):
    """
    Load the orbital data from a JSON file and return the information
    for a specific element based on its atomic number.

    Parameters:
    - json_file: the path to the JSON file containing the orbital data.
    - atomic_number: string, the atomic number of the element.

    Returns:
    - The orbital information for the given atomic number.
    """
    data = load_orbital_data(json_file)
    return get_orbital_info(data, atomic_number)

