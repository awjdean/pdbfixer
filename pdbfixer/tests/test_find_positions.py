from pathlib import Path

import numpy as np
import pytest
from openmm import Vec3, app, unit

import pdbfixer
from pdbfixer.pdbfixer import _overlayPoints


def _rotation(axis, angle):
    """Return the matrix that rotates by angle (in radians) around axis, which need not be a unit vector."""
    (x, y, z) = np.asarray(axis)/np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.identity(3) + np.sin(angle)*k + (1-np.cos(angle))*(k @ k)


@pytest.mark.parametrize("mirror", [np.identity(3), np.diag([1, 1, -1])], ids=["rotated", "mirrored"])
def test_overlayPointsRecoversRotation(mirror):
    """Test that _overlayPoints recovers a known rotation and translation, even when a reflection would fit better."""

    # The slab is centered, has principal axes x, y and z, and is thinnest along z, so the best proper rotation
    # from it onto itself, or onto its mirror image through the xy plane, is the identity.  Tilting the slab and
    # rotating its image by a known rotation makes that rotation the exact answer in both cases.  Moving the slab
    # by shift and its image by offset makes offset - known @ shift the exact translation.

    slab = [(1, 0, 0.1), (-1, 0, 0.1), (0, 2, -0.1), (0, -2, -0.1)]
    tilt = _rotation((1, -2, 0.5), 0.9)
    known = _rotation((1, 2, 3), 0.7)
    shift = np.array([0.3, 0.2, -0.4])
    offset = np.array([0.5, -1, 2])
    points2 = [Vec3(*(tilt @ p + shift)) for p in slab]
    points1 = [Vec3(*(known @ tilt @ mirror @ p + offset)) for p in slab]
    (rotate, translate) = _overlayPoints(points1, points2)
    assert np.allclose(rotate, known)
    assert np.allclose(translate, offset - known @ shift)


@pytest.mark.parametrize(
    "points1,points2,expected",
    [
        ([], [], (0, 0, 0)),
        ([Vec3(1, 2, 3)], [Vec3(4, 5, 6)], (-3, -3, -3)),
    ],
    ids=["none", "single"],
)
def test_overlayPointsTranslationOnly(points1, points2, expected):
    """Test that with fewer than two points, _overlayPoints returns a pure translation."""
    (rotate, translate) = _overlayPoints(points1, points2)
    assert np.array_equal(rotate, np.identity(3))
    assert np.allclose(translate, expected)


def test_overlayPointsTwoPoints():
    """Test that with two points, _overlayPoints maps each one exactly onto its partner with a proper rotation."""
    points2 = [Vec3(0, 0, 0), Vec3(1, 2, 2)]
    known = _rotation((1, 2, 3), 0.7)
    offset = np.array([0.5, -1, 2])
    points1 = [Vec3(*(known @ p + offset)) for p in points2]
    (rotate, translate) = _overlayPoints(points1, points2)
    assert np.allclose(rotate.T @ rotate, np.identity(3))
    assert np.isclose(np.linalg.det(rotate), 1)
    for target, point in zip(points1, points2):
        assert np.allclose(rotate @ point + translate, target)


def _load1BHL():
    """Load the 1BHL test structure."""
    return pdbfixer.PDBFixer(filename=(Path(__file__).parent / "data" / "1BHL.pdb").as_posix())


def _deleteAndRebuild(fixer, shouldDelete):
    """Delete the atoms selected by shouldDelete, then have PDBFixer add them back without building missing residues."""
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([atom for atom in modeller.topology.atoms() if shouldDelete(atom)])
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()


def _oxtPositions(fixer):
    """Return the positions of all OXT atoms."""
    return [pos for pos, atom in zip(fixer.positions, fixer.topology.atoms()) if atom.name == "OXT"]


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


def test_findOXTPosition():
    """Test that OXT is added at the correct position."""
    fixer = _load1BHL()

    # Record the original positions of the OXT atoms, then delete them and have PDBFixer add them back.

    originalPositions = _oxtPositions(fixer)
    assert len(originalPositions) > 0
    _deleteAndRebuild(fixer, lambda atom: atom.name == "OXT")

    # Make sure each one is sufficiently close to its original position.

    newPositions = _oxtPositions(fixer)
    assert len(newPositions) == len(originalPositions)
    for newPos, originalPos in zip(newPositions, originalPositions):
        assert unit.norm(newPos - originalPos) < 0.01 * unit.nanometer


def test_addedCBChirality():
    """Test that missing CB atoms are added with the correct chirality."""
    fixer = _load1BHL()

    # Record the handedness of every alanine, then delete its CB and have PDBFixer add it back.

    originalHandedness = _alanineHandedness(fixer)
    assert len(originalHandedness) > 0
    _deleteAndRebuild(fixer, lambda atom: atom.residue.name == "ALA" and atom.name == "CB")

    # Make sure none of them was built as its mirror image.  Adding atoms keeps the residues
    # in the same order, so the lists can be compared directly.

    assert _alanineHandedness(fixer) == originalHandedness
