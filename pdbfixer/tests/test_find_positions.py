from pathlib import Path

import numpy as np
from openmm import Vec3, app, unit

import pdbfixer
from pdbfixer.pdbfixer import _overlayPoints


def _alanineHandedness(fixer):
    """Return the handedness (+1 or -1) of every alanine's alpha carbon, in residue order."""
    positions = fixer.positions.value_in_unit(unit.nanometer)
    handedness = []
    for residue in fixer.topology.residues():
        if residue.name == "ALA":
            pos = {atom.name: positions[atom.index] for atom in residue.atoms()}
            toN = pos["N"]-pos["CA"]
            toC = pos["C"]-pos["CA"]
            toCB = pos["CB"]-pos["CA"]
            handedness.append(np.sign(np.dot(toN, np.cross(toC, toCB))))
    return handedness


def test_overlayPointsNeverReflects():
    """Test that _overlayPoints returns a proper rotation even when a reflection would fit better."""
    points2 = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(1, 1, 0.1)]
    points1 = [Vec3(p[0], p[1], -p[2]) for p in points2]
    (_, rotate, _) = _overlayPoints(points1, points2)
    assert np.isclose(np.linalg.det(rotate), 1)


def test_findOXTPosition():
    """Test that OXT is added at the correct position."""
    fixer = pdbfixer.PDBFixer(filename=(Path(__file__).parent / "data" / "1BHL.pdb").as_posix())

    # Record the original position of OXT, then delete it.

    originalPos = [pos for pos, atom in zip(fixer.positions, fixer.topology.atoms()) if atom.name == "OXT"]
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([atom for atom in modeller.topology.atoms() if atom.name == "OXT"])
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    # Have PDBFixer add it back and make sure it is sufficiently close to the original position.

    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    newPos = [pos for pos, atom in zip(fixer.positions, fixer.topology.atoms()) if atom.name == "OXT"]
    assert unit.norm(newPos[0] - originalPos[0]) < 0.01 * unit.nanometer


def test_addedCBChirality():
    """Test that missing CB atoms are added with the correct chirality."""
    fixer = pdbfixer.PDBFixer(filename=(Path(__file__).parent / "data" / "1BHL.pdb").as_posix())

    # Record the handedness of every alanine, then delete its CB.

    originalHandedness = _alanineHandedness(fixer)
    assert len(originalHandedness) > 0
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([atom for atom in modeller.topology.atoms() if atom.residue.name == "ALA" and atom.name == "CB"])
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    # Have PDBFixer add them back and make sure none of them was built as its mirror image.
    # Adding atoms keeps the residues in the same order, so the lists can be compared directly.

    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    assert _alanineHandedness(fixer) == originalHandedness
