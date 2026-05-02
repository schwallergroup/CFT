from ase.io import read, write
from pathlib import Path
import os
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.mep import NEB, SingleCalculatorNEB
from mace.calculators import mace_mp

MODEL_PATH = os.environ.get('MODEL_PATH', '')  # set MODEL_PATH env var to your MACE model file
MODEL_HEAD = "matpes_r2scan"
DEVICE = "cuda"

def relax_single_atom(atoms_obj, atom_symbol, model_path, head, device):
    """Relax a single atom type on a structure with all others except the target atom fixed."""
    # Keep target atom and metal, remove the other adsorbate
    other_adsorbate = 'O' if atom_symbol == 'C' else 'C'
    subset = atoms_obj[[atom.index for atom in atoms_obj if atom.symbol != other_adsorbate]]
    fixed = FixAtoms(indices=[atom.index for atom in subset if atom.symbol != atom_symbol])
    subset.set_constraint(fixed)
    subset.calc = mace_mp(model=model_path, head=head, device=device)
    opt = BFGS(subset)
    opt.run(fmax=0.03, steps=300)
    return subset[[atom.index for atom in subset if atom.symbol == atom_symbol]][0].position

def main(file, name):
    images = read(file, index=':')
    assert len(images) == 3

    is_image = images[0]
    pmd_image = images[1]
    fs_image = images[2]

    for a in images:
        fixed_idx = [atom.index for atom in a if atom.symbol not in ['C', 'O']]
        constraint = FixAtoms(indices=fixed_idx)
        a.set_constraint(constraint)

    # Relax is_image: Remove O relax C on particle, then remove C relax O on particle
    c_pos = relax_single_atom(is_image, 'C', MODEL_PATH, MODEL_HEAD, DEVICE)
    o_pos = relax_single_atom(is_image, 'O', MODEL_PATH, MODEL_HEAD, DEVICE)
    
    # Copy relaxed C and O positions to is_image
    for atom in is_image:
        if atom.symbol == 'C':
            atom.position = c_pos
        elif atom.symbol == 'O':
            atom.position = o_pos

    # Relax is_image normally
    is_image.calc = mace_mp(model=MODEL_PATH, head=MODEL_HEAD, device=DEVICE)
    opt = BFGS(is_image)
    opt.run(fmax=0.03, steps=300)
    
    # Relax fs_image normally
    fs_image.calc = mace_mp(model=MODEL_PATH, head=MODEL_HEAD, device=DEVICE)
    opt = BFGS(fs_image)
    opt.run(fmax=0.03, steps=300)

    neb_images = [is_image]
    for _ in range(n_images0):
        img = is_image.copy()
        neb_images.append(img)
    neb_images.append(pmd_image)

    _neb0 = NEB(neb_images)
    _neb0.interpolate(apply_constraint=True)


    neb_images = [pmd_image]
    for _ in range(n_images1):
        img = pmd_image.copy()
        neb_images.append(img)
    neb_images.append(fs_image)

    _neb1 = NEB(neb_images)
    _neb1.interpolate(apply_constraint=True)

    # Combine chains, skipping duplicate pmd_image at junction
    images = _neb0.images + _neb1.images[1:]
    
    # Add metadata about source to each image
    for i, img in enumerate(images):
        if i < len(_neb0.images):
            img.info['source'] = 'NEB_chain_0_is_to_pmd'
            img.info['chain_index'] = i
        else:
            img.info['source'] = 'NEB_chain_1_pmd_to_fs'
            img.info['chain_index'] = i - len(_neb0.images)
        img.info['file'] = file
        img.info['total_images'] = len(images)
    
    for a in images:
        a.calc = mace_mp(model=MODEL_PATH, head=MODEL_HEAD, device=DEVICE)


    neb = SingleCalculatorNEB(images)
    opt = FIRE(neb, trajectory=f'OUT_{name}.traj')
    opt.run(fmax=0.05, steps=300)
    
    # Evaluate energies on edge points to ensure calculator is attached
    images[0].get_potential_energy()
    images[-1].get_potential_energy()
    
    # Write final converged chain to XYZ
    write(f'OUT_{name}_final.xyz', images)


n_images0 = 5
n_images1 = 3


if __name__ == '__main__':

    for i in range(65):

        file = f'./NEB_seeds/NEB_{i}.xyz'
        name = f'neb_{i}'
        start_file = f"RUN_{name}.START"
        path = Path(start_file)

        if not path.exists():
            path.write_text(start_file)

        else:
            print(f'Found start file: {start_file}. . . SKIPPING.')
            continue

        print(f'STARTING: {name}. . .')
        
        main(file, name)  
    
