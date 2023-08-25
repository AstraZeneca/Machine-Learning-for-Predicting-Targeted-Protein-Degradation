import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys

from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

def get_fingerprint(smiles: str,
                    n_bits: int = 1024,
                    fp_type: Literal['morgan', 'maccs', 'path'] = 'morgan',
                    min_path: int = 1,
                    max_path: int = 2,
                    atomic_radius: int = 2) -> np.ndarray:
    """Returns molecular fingerprint of a given molecule SMILES.

    Args:
        smiles (str): SMILES string to convert.
        n_bits (int, optional): Number of bits of the generated fingerprint. Defaults to 1024.
        fp_type (Literal[&#39;morgan&#39;, &#39;maccs&#39;, &#39;path&#39;], optional): Fingerprint type to generate. Defaults to 'morgan'.
        min_path (int, optional): Minimum path lenght for path-based fingerprints. Defaults to 1.
        max_path (int, optional): Maximum path lenght for path-based fingerprints. Defaults to 2.
        atomic_radius (int, optional): Atomic radius for MORGAN fingerprints. Defaults to 2.

    Raises:
        ValueError: When wrong fingerprint type is requested.

    Returns:
        np.ndarray: The generated fingerprint.
    """ 
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == 'morgan':
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, atomic_radius,
                                                            nBits=n_bits)
    elif fp_type == 'maccs':
        fingerprint = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == 'path':
        fingerprint = Chem.rdmolops.RDKFingerprint(mol, fpSize=n_bits,
                                                   minPath=min_path,
                                                   maxPath=max_path)
    else:
        raise ValueError(f'Wrong type of fingerprint requested. Received "{fp_type}", expected one in: [morgan|maccs|path]')
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array
