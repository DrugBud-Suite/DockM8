#!/usr/bin/env python2.7
# -*- coding: UTF-8 -*-

"""
 title: panther.py
 description: A tool for creating negative image representations of protein
              binding pockets
 author: Kari Salokas (kari.salokas@gmail.com) 2014-2016
 modifications: Mira Ahinko 2016-2018
 from: Computational Bioscience Laboratory (http://www.jyu.fi/motu/cbl/);
       University of Jyvaskyla, Finland and University of Eastern Finland,
       Finland
 version: 0.18.21
 usage: python panther.py [options] input.txt outputfile.mol2
 notes:
 python_version: 2.7.5

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""



import sys
import math
import re
from collections import defaultdict
from time import time
from itertools import product
from itertools import combinations
from copy import copy
from textwrap import TextWrapper
from os import path



"""
TODO list:
- code cleaning, commenting and optimizing. Currently the program is littered
  with unnecessary calculations and suboptimal solutions. It also is in dire
  need of comments.
- Testing waters more thoroughly
- All string comparisons should be made in lowercase only. Input should be
  changed to lowercase right away, and unnecessary conversions could be deleted
  after that. Elements and the like should then only be changed to uppercase
  when output is generated.
- more debugging...
- Amino acids should be migrated over to angles.lib and made to comply with
  that format.
- Shorten lines
"""

# Class for storing all info about an atom, as well as a few helpful variables.
class Atom(object):

    def __init__(self, renam, idn, name, altloc, resname, chainid, resseq, icode, x, y, z, occup, tempf, elemen, charge):
        self.renam = renam
        self.idn = int(idn)
        self.name = str(name).strip()

        self.resname = resname.strip()
        if (len(chainid.strip()) < 1):
            chainid = "X"
        self.chainid = chainid

        self.resseq = 0
        if resseq is not None and len(str(resseq)) > 0:
            self.resseq = int(resseq)

        ax = "%.3f" % float(x)
        self.x = float(ax)
        ay = "%.3f" % float(y)
        self.y = float(ay)
        az = "%.3f" % float(z)
        self.z = float(az)

        self.tempf = float(tempf)
        self.elemen = elemen.strip()
        if len(self.elemen) > 1:
            newe = ""
            for a in self.elemen:
                if not a.isdigit():
                    newe += a
            if len(newe) > 1:
                if newe.lower().startswith("h"):
                    if newe.lower() == "ho":
                        newe = "h"
            self.elemen = newe
            #~ self.elemen = self.elemen[0]
        self.safe = False

        # These attributes are not used, but need to be initialized for cloning.
        self.altloc = altloc
        self.icode = icode
        self.occup = occup
        self.charge = charge

        # Additional variables are initialized immediately to help remember what
        # has already been implemented.
        self.distancetocenter = 1000
        self.radius = 2.0
        self.bonds = []
        self.myAA = []
        self.anglesList = defaultdict(list)
        self.close = []
        self.new_ch = 0.0
        self.addline = ""
        self.kiddies = []
        self.host = None
        self.donor_bonds = set()
        self.acceptor_bonds = set()
        self.testchainid = "a"
        self.testresseq = 1
        self.testresname = "na"
        self.hasmods = False
        self.bonds_to_h = []
        self.new_bonds = []
        self.ligbond = None
        # ugly solution for a problem that hasn't been fixed yet. remove once
        # done.
        self.new_H_bonds = []

    def __copy__(self):
        newatom = Atom(self.renam, self.idn, self.name, self.altloc, self.resname, self.chainid, self.resseq, self.icode, self.x, self.y, self.z, self.occup, self.tempf, self.elemen, self.charge)
        newatom.altloc = self.altloc
        newatom.icode = self.icode
        newatom.occup = self.occup
        newatom.charge = self.charge

        # Additional variables are initialized immediately to help remember what
        # has already been implemented.
        newatom.distancetocenter = self.distancetocenter
        newatom.radius = self.radius
        newatom.bonds = self.bonds
        newatom.myAA = self.myAA
        newatom.myAA = self.anglesList
        newatom.close = self.close
        newatom.new_ch = self.new_ch
        newatom.addline = self.addline
        newatom.kiddies = self.kiddies
        newatom.host = self.host
        newatom.donor_bonds = self.donor_bonds
        newatom.acceptor_bonds = self.acceptor_bonds
        newatom.testchainid = self.testchainid
        newatom.testresseq = self.testresseq
        newatom.testresname = self.testresname
        newatom.hasmods = self.hasmods
        newatom.bonds_to_h = self.bonds_to_h
        newatom.new_bonds = self.new_bonds
        newatom.ligbond = self.ligbond
        newatom.new_H_bonds = self.new_H_bonds
        return newatom


class Taskutus:

    def __init__(self):
        # TODO: Add options as object variables.
        self.donor_groups = {"ne": ["he"], "nh1": ["1hh1"], "nh2": ["1hh2"], # Arg
                        "od2": ["hd2"], # Ash
                        "nd2": ["hd21"], # Asn
                        "oe2": ["he2"], # Glh
                        "ne2": ["he21", "he2"], # Gln, Hie/Hip
                        "nd1": ["hd1"], # Hid/Hip/His
                        "nz": ["hz2"], # Lys/Lyn
                        "og": ["hg"], # Ser
                        "og1": ["hg1"], # Thr
                        "ne1": ["he1"], # Trp
                        "oh": ["hh"] # Tyr
                        }


    # Key functions. Could more clearly be implemented with lambda functions.

    # Custom key for arranging atom input modifiers. shorter names = applied first.
    def atomfieldkey(self, entry1, entry2):
        if len(entry1[0]) < len(entry2[0]):
            return -1
        elif len(entry1[0]) > len(entry2[0]):
            return 1
        else:
            return 0

    # Custom key for sorting based on distance from center atom
    def centerkey(self, a, b):
        if float(a.distancetocenter) > float(b.distancetocenter):
            return -1
        elif float(a.distancetocenter) < float(b.distancetocenter):
            return 1
        else:
            return 0

    # Custom key for arranging atoms in order based on amino acid sequence number
    def ownAAkey(self, entry1, entry2):
        if entry2[0].resseq == 0 or entry1[0].resseq < entry2[0].resseq:
            return -1
        elif entry1[0].resseq == 0 or entry1[0].resseq > entry2[0].resseq:
            return 1
        else:
            return 0

    # returns an atom that is le from frompoint towards topoint
    def get_to_from(self, toPoint, fromPoint, distance, atomrad):
        d = self.distance(toPoint, fromPoint)
        r = distance / d
        x = toPoint.x - fromPoint.x
        y = toPoint.y - fromPoint.y
        z = toPoint.z - fromPoint.z
        x = x * r + fromPoint.x
        y = y * r + fromPoint.y
        z = z * r + fromPoint.z
        newAtom = Atom("null", 0, "Br", "", "FIL", "", "0", "", x, y, z,
                  "", "0.000", "Br", "0")
        newAtom.radius = atomrad
        return newAtom

    # creates a ray of atoms between two points, with increment between each
    # point of the ray.
    def createray(self, start, finish, increment, atomradius):
        d = self.distance(start, finish)
        ray = []
        i = increment
        while i < d:
            ray.append(self.get_to_from(start, finish, i, atomradius))
            i = i + increment
        return ray

    # rounds a number. returns an int.
    def roundit(self, x, base=20):
        return int(base * round(float(x)/base))

    # returns the largest distance between pairs of closest atoms in the list.
    def get_largest_closest_distance(self, atomlist):
        ret = 0.0
        closest_distance = 10000.0
        for a in atomlist:
            for b in atomlist:
                d = self.distance(a, b)
                # Compare d instead of atom.idn as the idn can be the same..
                if d > 0.0001 and d < closest_distance:
                    closest_distance = d
        if closest_distance < 9999.0 and closest_distance > ret:
            ret = closest_distance
        return ret

    # returns atoms that might be lining the cavity in which centers are.
    def get_lining(self, centers, protein):
        #lining = defaultdict(list)
        ret = []

        abc="abcdefghijkl"
        current = 0
        curres = 1
        testlist = []

        for c in centers:
            sectors = list(c.anglesList.values())
            for sec in sectors:
                curres += 1
                for atom in sec:
                    atom.testchainid = abc[current%len(abc)]
                    atom.testresseq = curres
                    atom.testresname = "sec"
                    testlist.append(atom)
                prime = self.get_closest(c, sec)[0]
                ret.append(prime)
            curres = 0
            current += 1
        # deletes possible duplicate entries.
        return [list(set(ret)), testlist]

    # Adds angle information about every atom to each center. The atoms will be
    # divided into sectors, size of which is defined by ang.
    def dovecs(self, protein, centers, refpoint1, refpoint2, refpoint3, ang):
        for c in centers:
            refvec1 = self.getvec(refpoint1, c)
            refvec1 = self.getvec(refpoint1, c)
            refvec2 = self.getvec(refpoint2, c)
            refvec3 = self.getvec(refpoint3, c)

            for p in protein:
                if not c.idn == p.idn:
                    vecp = self.getvec(p, c)

                    refangle1 = self.roundit(self.angle(vecp, refvec1), base=ang)
                    refangle2 = self.roundit(self.angle(vecp, refvec2), base=ang)
                    refangle3 = self.roundit(self.angle(vecp, refvec3), base=ang)
                    c.anglesList[refangle1, refangle2, refangle3].append(p)


    # returns a vector perpendicular to the given vector.
    def get_perpendicular(self, vector, seed):
        x2 = 1.43*seed
        y2 = -1.5*seed
        for i in range(0, len(vector)):  # TODO: better way to do this.
            if vector[i] == 0.0:
                vector[i] = 0.0001
        z2 = (0-(vector[0]*x2)-(vector[1]*y2))/vector[2]
        return [x2, y2, z2]

    # returns a circle that has all its atoms at given angle of atom and
    # angle_atom. distance = distance from atom to circle. angle_increment in
    # degrees
    def get_circle(self, main_atom, angle_atom, angle, distance, angle_increment, atom_radius):
        if (angle == 180):
            ret = self.get_to_from(main_atom, angle_atom, self.distance(angle_atom, main_atom) + distance, atom_radius)
            ret.elemen = "Mn"
            ret.name = "Mn"
            return [ret]
        if (angle == 0):
            self.print_this("err", "0 Angle in circle generation!")
            return [None]

        # Center of the circle:
        dist_to_circle_center = math.cos(math.radians(angle)) * -distance
        circle_center = self.get_to_from(main_atom, angle_atom, self.distance(main_atom, angle_atom) + dist_to_circle_center, atom_radius)


        # Unit vectors:
        axel_vector = self.get_unit_vector(self.getvec(angle_atom, main_atom))
        perpendicular1 = self.get_unit_vector(self.get_perpendicular(axel_vector, 1))
        perpendicular2 = self.get_unit_vector(self.get_second_perpendicular(axel_vector, perpendicular1))

        # The circle itself:
        circle_radius = math.tan(math.radians(angle)) * dist_to_circle_center
        circle = []
        ang = 0
        increment = ((2*math.pi)/360) * angle_increment
        i = 0
        while ang < 2*math.pi - 0.5 * increment:
            x = circle_center.x + (circle_radius*math.cos(ang)*perpendicular1[0]) + (circle_radius*math.sin(ang)*perpendicular2[0])
            y = circle_center.y + (circle_radius*math.cos(ang)*perpendicular1[1]) + (circle_radius*math.sin(ang)*perpendicular2[1])
            z = circle_center.z + (circle_radius*math.cos(ang)*perpendicular1[2]) + (circle_radius*math.sin(ang)*perpendicular2[2])
            ang = ang + increment
            newAtom = Atom("null", i, "Mg", "", "CIR", "", 55, "", x, y, z,"", "0.000", "Mg", "0")
            newAtom.radius = atom_radius # 0.1
            i = i + 1
            newAtom.name = main_atom.name
            circle.append(newAtom)
        return circle

    # returns vector length
    def vector_length(self, vector):
        temp = 0
        for num in vector:
            temp = temp + (num*num)
        return math.sqrt(temp)

    # returns a unit vector from the given vector.
    def get_unit_vector(self, vector):
        leng = self.vector_length(vector)
        newvec = []
        for num in vector:
            newvec.append(num/leng)
        return newvec

    # returns a vector perpendicular to both given vectors.
    def get_second_perpendicular(self, v1, v2):
        x = (v1[1]*v2[2]) - (v1[2]*v2[1])
        y = (v1[2]*v2[0]) - (v1[0]*v2[2])
        z = (v1[0]*v2[1]) - (v1[1]*v2[0])
        return [x, y, z]

    # connects residues to one another, forming a chain.
    def connect_residues(self, amino_dict, protein):
        for key, value in amino_dict.items():
            for atom in value:
                if atom.name.lower() == "n" and key > 0:
                    for atom2 in amino_dict[key - 1]:
                        if atom2.name.lower() == "c":
                            cand = atom2
                            if (self.distance(atom, atom2) > 2.0):
                                best = []
                                for atom3 in protein:
                                    dist = self.distance(atom, atom3)
                                    if (dist < 2.0):
                                        if atom3.name.lower() == "c":
                                            best.append([atom3, dist])
                                best.sort(key = lambda x: x[1])
                                if( len(best) < 1):
                                    best = []
                                    circle = self.get_circle(atom, atom.bonds[0], 110, 1.3, 5, 0.0)
                                    for c in circle:
                                        best.append([c, self.distance(c, atom.bonds[0])])
                                    best.sort(key = lambda x: x[1])
                                cand = best[0][0]
                            atom.bonds.append(cand)
                            break

    # removes all atoms except those with a matching resid
    def delete_AAid(self, atoms, ids):
        ret = []
        for a in atoms:
            if a.resseq not in ids:
                ret.append(a)
        return ret

    # returns the minimal distance between two atom lists.
    def set_distance(self, atomlist1, atomlist2):
        distance = 1000
        for a in atomlist1:
            for a2 in atomlist2:
                dist = self.distance(a, a2)
                if (dist < distance):
                    distance = dist
        return distance

    # returns True, if a given atom has a hydrogen atom bonded to it.
    def has_h(self, atom):
        ret = False
        h = []
        for a in atom.bonds:
            if a.elemen.lower() == "h":
                ret = True
                h.append(a)
        return [ret, h]

    # returns None if list is empty. Otherwise smallest distance from atom to
    # any atom in the given list.
    def distance_from_list(self, atom, list_of_atoms):
        if len(list_of_atoms) < 1:
            return None
        current = self.distance(atom, list_of_atoms[0])
        for a in list_of_atoms:
            if self.distance(atom, a) < current:
                current = self.distance(atom, a)
        return current


    # Create coordination points for two-hydrogen waters, return as an atom list.
    def check_waters(self, waters, prot, lig, angle_dict, chargerad, filler_rad, h_bond_distance, H_add, cir_cheat={}, angle_tolerance=35, bond_no=2, dist=3.5):
        charged_prot = []
        charged_lig = []

        waters = list(waters)

        for p in prot:
            # Skip hydrogens for now and only check them later.
            # If charged, add to list of charged atoms.
            if (p.elemen.lower() != "h" and self.is_charged(p) and
            self.distance_from_list(p, waters) < dist):
                charged_prot.append(p)

        for l in lig:
            # If charged, add to list of charged atoms
            if self.is_charged(l) and self.distance_from_list(l, waters) < dist:
                charged_lig.append(l)

        water_oxy = []
        # First remove all H atoms from the waters list. Or create a new list
        # without them, in other words
        for wat in waters:
            if wat.elemen.lower() == "o":
                water_oxy.append(wat)

        two_h = []
        one_h = []
        no_h = []

        # Sort water oxygens into three lists: those with two H atoms bound to
        # them, those with one and those with none
        for wato in water_oxy:
            test = self.has_h(wato)
            if len(test[1]) > 1:
                two_h.append(wato)
            elif len(test[1]) == 1:
                one_h.append(wato)
            else:
                no_h.append(wato)
        if len(no_h) > 0:
            self.print_this(
                "war",
                "Water oxygens without hydrogens found.\n" +
                "  - For now, coordination points are not added for those.")
        if len(one_h) > 0:
            self.print_this(
                "war",
                "Water oxygens with just one hydrogen found.\n" +
                "  - For now, coordination points are not added for those.")

        # a list for charged spots
        ch_spots = []
        # a list for water molecules that charged spots can be generated for.
        #Will contain lists as entries in the form of [water_atom, ch_spots]
        has_chs = []

        for whole_wat in two_h:
            circ = self.get_circle(whole_wat, whole_wat.bonds[0], 109.5, h_bond_distance, 5, chargerad)
            # Create a tetrahedral base of three atoms based on main atom
            # (oxygen) and one hydrogen (which will be one of the angles for now)
            tetra = self.get_tetra(whole_wat, circ, whole_wat.bonds[1])
            acc_spots = []
            # Make sure the tetra is sorted in a way that puts the atom we don't
            # want first.
            tetra.sort(key=lambda x: self.distance(x, whole_wat.bonds[1]))

            # And now put them into the acceptor list
            for tet_at in tetra[1:]:
                    acc_spots.append(tet_at)
                    tet_at.resseq = whole_wat.resseq
                    tet_at.resname = whole_wat.resname
                    tet_at.tempf = -whole_wat.new_ch

            whole_wat_ch_spots = acc_spots

            # And now we'll do spots for the two hydrogens
            for hydrogen in whole_wat.bonds:
                donspot = self.get_to_from(hydrogen, whole_wat, h_bond_distance+H_add, chargerad)
                donspot.resseq = whole_wat.resseq
                donspot.resname = whole_wat.resname
                donspot.tempf = -hydrogen.new_ch
                whole_wat_ch_spots.append(donspot)

            has_chs.append([whole_wat, whole_wat_ch_spots])

            ch_spots.extend(whole_wat_ch_spots)

        # resnames with only acceptor oxygens.
        acc_o = ["gln", "asn"]
        # list for oxygens that are assumed to be acceptors unless their H is
        # between the water atom and oxygen
        check_o = ["asp", "glu"]

        # TODO: Fix/finish/rewrite and iterate over one_h.
        # TODO: Check angles. Fix donor/acceptor logic. Other things necessary.
        for part_wat in []:
            don_spots = []
            acc_spots = []
            # Get a point beyond the one bound H atom that reflects a donor spot.
            don_spots.append(self.get_to_from(part_wat.bonds[0], part_wat, h_bond_distance+H_add, chargerad))

            # And now go over the protein + ligand list and see if we can spot
            # some easy acceptors or donors.
            #candidates = []
            #secondary_candidates = []

            for ch_p in charged_prot:
                # Check that the atom in question is close enough to be considered.
                if self.distance(ch_p, part_wat) > dist:
                    continue
                if ch_p.elemen.lower() == "n":
                    if ch_p.name.lower() == "n":
                        acc_spots.append(ch_p)
                    elif not ch_p.name.lower() == "nd" and ch_p.resname.lower() == "hie":
                        if not ch_p.name.lower() == "ne" and ch_p.resname.lower() == "hid":
                            acc_spots.append(ch_p)
                elif ch_p.elemen.lower() == "o":
                    if ch_p.name.lower() == "o":
                        don_spots.append(ch_p)
                    elif ch_p.resname in acc_o:
                        don_spots.append(ch_p)
                    elif ch_p.resname in check_o:
                        hydr = None
                        for bond_atom in ch_p.bonds:
                            if bond_atom.elemen.lower() == "h":
                                hydr = bond_atom
                        if hydr is not None:
                            if self.distance(hydr, part_wat) < self.distance(ch_p, part_wat):
                                acc_spots.append(ch_p)
                            else:
                                don_spots.append(ch_p)

            # Now we've dug around the immediate areas for acceptor or donor
            # spots provided by the protein structure. Time to do the same for
            # the ligand, IF needed
            # TODO: check spots from angles.lib
            #if len(don_spots) < 2 or len(acc_spots) < 2:
            #    for ch_p in charged_lig:


            # If we still don't have enough, it's time to get desperate and
            # check the charged spots of other H2O molecules.
            if len(don_spots) < 2 or len(acc_spots) < 2:
                for entry in has_chs:
                    main_atom = entry[0]
                    ch_hypotheticals = entry[1]
                    # If the main atom (another water molecule's oxygen) is
                    # close enough, we'll check it out
                    if self.distance(main_atom, part_wat) < dist:
                        for spotfulness in ch_hypotheticals:
                            # If the spot is closer to our water atom than its
                            # own host atom. in other words, if it's between the
                            # two examined oxygens OR close to the oxygen we're
                            # trying to get done
                            if self.distance(spotfulness, part_wat) < self.distance(part_wat, main_atom):
                                # negative charge -> we'll be accepting
                                # a hydrogen.
                                if spotfulness.tempf < 0:
                                    valid = True
                                    for don_hypo in don_spots:
                                        # IF there is a confirmed donor spot
                                        #BETWEEN this new acceptor spot and our
                                        # oxygen, we can't use it.
                                        if self.distance(don_hypo, spotfulness) < self.distance(part_wat, spotfulness):
                                            valid = False
                                    if valid:
                                        acc_spots.append(self.get_to_from(spotfulness, part_wat, h_bond_distance, chargerad))
                                # Positive charge -> we'll be donating one.
                                elif spotfulness.tempf > 0:
                                    valid = True
                                    for acc_hypo in acc_spots:
                                        # IF there is a confirmed acceptor spot
                                        # BETWEEN this new donor spot and our
                                        # oxygen, we can't use it.
                                        if self.distance(acc_hypo, spotfulness) < self.distance(part_wat, spotfulness):
                                            valid = False
                                    if valid:
                                        don_spots.append(self.get_to_from(spotfulness, part_wat, h_bond_distance+H_add, chargerad))

            if len(don_spots) >= 2:
                circ = self.get_circle(part_wat, don_spots[0], 109.5, h_bond_distance, 5, chargerad)
                # Create a tetrahedral base of three atoms based on main atom
                # (oxygen) and one hydrogen (which will be one of the angles
                # for now)
                tetra = self.get_tetra(part_wat, circ, don_spots[1])
                new_acc_spots = []
                # Parse only the two acceptor spots out of the three-atomed
                # tetrahedral base.
                for tet_at in tetra:
                    if self.no_overlap(tet_at, don_spots)[0]:
                        new_acc_spots.append(tet_at)
                        tet_at.tempf = -part_wat.new_ch
                        tet_at.resseq = part_wat.resseq
                        tet_at.resname = part_wat.resname
                ch_spots.extend(new_acc_spots)
                has_chs.append([part_wat, new_acc_spots])

            elif len(acc_spots) >= 2:
                circ = self.get_circle(part_wat, acc_spots[0], 109.5, h_bond_distance+H_add, 5, chargerad)
                tetra = self.get_tetra(part_wat, circ, don_spots[1])
                new_don_spots = []
                for tet_at in tetra:
                    if self.no_overlap(tet_at, acc_spots)[0]:
                        new_don_spots.append(tet_at)

                        close_h = None
                        # Figure out what the charge of the hypothetical spot
                        # should be
                        for a in part_wat.bonds:
                            if a.elemen.lower() == "h" and self.distance(a, tet_at) < self.distance(part_wat, tet_at):
                                close_h = a
                        if close_h is None:
                            assigned_charge = part_wat.new_ch
                        else:
                            assigned_charge = -close_h.new_ch
                        tet_at.tempf = assigned_charge
                        tet_at.resseq = part_wat.resseq
                        tet_at.resname = part_wat.resname

                        ch_spots.extend(new_don_spots)
                        has_chs.append([part_wat, new_don_spots])

        finished_charged_spots = []
        overlap_check = []
        for line in has_chs:
            overlap_check.append(line[0])
            for bond_atom in line[0].bonds:
                overlap_check.append(bond_atom)

        # Lastly, its time to check for overlaps.
        for spot_atom in ch_spots:
            add = True
            for water_atom in overlap_check:
                if self.distance(water_atom, spot_atom) < (spot_atom.radius+water_atom.radius):
                    add = False
                    break
            if add:
                finished_charged_spots.append(spot_atom)

        return finished_charged_spots

    # returns the biggest or smallest residue that does not contain metal
    def get_bs_non_metal(self, atomlist, metals, not_these, biggest):
        ret = [[], 0]
        for key, value in atomlist.items():
            if not value[0].resname.lower() in not_these:
                metal = False
                for at in value:
                    if at.elemen in metals:
                        metal = True
                val_len = len(value)
                if (not biggest):
                    val_len = -val_len
                if not metal and val_len > ret[1]:
                    ret = [value, len(value)]
        return ret[0]

    # like above, but containing metal.
    def get_bs_metal(self, atomlist, metals, biggest):
        ret = [[], 0]
        for key, value in atomlist.items():
            metal = False
            for at in value:
                if at.elemen in metals:
                    metal = True
            val_len = len(value)
            if (not biggest):
                val_len = -val_len
            if metal and val_len > ret[1]:
                ret = [value, len(value)]
        return ret[0]

    # builds water molecules out of water atoms. in other words, adds entries
    # to atom.bonds and atom.myAA -lists
    def build_wat(self, water, covrad, restol):
        new_water_list = []
        for at in water:
            new_water_list.append([at.resseq, [at]])
        new_water_list.sort(key=lambda x: x[0])

        molecule_list = []
        for wat_entry in new_water_list:
            if len(molecule_list) < 1:
                molecule_list.append(wat_entry)
            else:
                found = False
                for line in molecule_list:
                    if line[0] == wat_entry[0]:
                        line[1].extend(wat_entry[1])
                        found = True
                        break
                if not found:
                    molecule_list.append(wat_entry)

        for entry in molecule_list:
            for atumba in entry[1]:
                for atumba2 in entry[1]:
                    if atumba is not atumba2:
                        atumba.bonds.append(atumba2)
                        atumba.myAA.append(atumba2)

    # returns three atoms with highest values of x, y and z
    def get_highest(self, allatoms):

        highx = allatoms[0]
        highy = allatoms[0]
        highz = allatoms[0]
        for atom in allatoms:
            if atom.x > highx.x:
                highx = atom
            if atom.y > highy.y:
                highy = atom
            if atom.z > highz.z:
                highz = atom
        return [highx, highy, highz]


    def findcavity(self, iputdata): # The main program.
        # Preparation:
        self.setradi(iputdata["allatoms"], iputdata["distanceinfo"])


        refx, refy, refz = self.get_highest(iputdata["allatoms"])

        refpoint1 = Atom("null", 0, "O", "", "REF", "", 0, "", refx.x + 10, refx.y - 5, refx.z,"", "0.000", "O.3", "0")
        refpoint2 = Atom("null", 0, "O", "", "REF", "", 0, "", refy.x - 5, refy.y + 10, refy.z,"", "0.000", "O.3", "0")
        refpoint3 = Atom("null", 0, "O", "", "REF", "", 0, "", refz.x, refz.y - 5, refz.z,"", "0.000", "O.3", "0")


        self.dovecs(iputdata["protatoms"], iputdata["centers"], refpoint1, refpoint2, refpoint3, iputdata["linangle"])


        if (iputdata["centers_only"]):
            self.print_this("inf", "Outputting center atoms...")
            return [["centers", iputdata["centers"], None]]

        # Identify the pocket lining:
        lining, sectorstest = self.get_lining(iputdata["centers"], iputdata["protatoms"])

        if iputdata["sectortest"]:
            for atom in sectorstest:
                atom.resseq = atom.testresseq
                atom.resname = atom.testresname
                atom.chainid = atom.testchainid
            return [["Sectors", sectorstest, None]]

        if (not self.is_null(iputdata["force_lining"][0])):
            lining.extend(self.get_AAs(iputdata["force_lining"], iputdata["protatoms"], chain=True))

        # Remove amino acids to be ignored, if so specified by the user.
        if (not self.is_null(iputdata["ignore_lining"][0])):
            ignore_set = set()
            for id_num in iputdata["ignore_lining"]:
                ignore_set.add(int(id_num))
            lining = self.delete_AAid(lining, ignore_set)


        self.print_this("ver-1", "Using " + str(len(iputdata["centers"])) + " centers for lining calculations.")
        # Full amino acids, instead of just individual atoms
        if(iputdata["getFull"]):
            # using a set just in case two atoms of the same amino acid are
            # found in lining.
            new_list = set()
            for atom in lining:
                for atom2 in iputdata["AA_original_numbers"][atom.chainid + "-" + str(atom.resseq)]:
                    new_list.add(atom2)
            lining = list(new_list)
        self.print_this("ver-1", str(len(lining)) + " possible lining atoms found. Refining... ")

        # if adjacent amino acids should be taken into account, they first need
        # to be fetched.
        if(iputdata["getAdjacent"]):
            lining = self.withinRad(list(lining), iputdata["protatoms"], iputdata["AAradius"])
            new_list = set()
            for atom in lining:
                for atom2 in iputdata["AA_original_numbers"][atom.chainid + "-" + str(atom.resseq)]:
                    new_list.add(atom2)
            lining = list(new_list)
            self.print_this("ver-1", "Lining expanded to " + str(len(lining)) + " atoms")

        centeravg = self.get_avg_point(iputdata["centers"])

        # If more than one chain is present, we'll only consider the ones that
        #  are connected to the closest chain. in other words, the chains that
        # have <safe_limit ångstroms between them and the closest chain.
        if (len(iputdata["chains_dict"]) > 1):

            chains = []
            lining.sort(key=lambda x: x.chainid)
            c = [lining[0]]
            for l in lining:
                if l.chainid == c[0].chainid:
                    c.append(l)
                else:
                    chains.append(c)
                    c = [l]
            chains.append(c)
            save_chain = self.get_closest(centeravg, lining)[0].chainid
            ref_chain = []
            for c in chains:
                if c[0].chainid == save_chain:
                    ref_chain = c
                    break
            safe = []
            safe_limit = 8
            for c in chains:
                if (self.set_distance(c, ref_chain) < safe_limit):
                    safe.append(c)

            lining = []
            for s_c in safe:
                lining.extend(s_c)

        # Only qualify connected lining atoms. If creep is used.
        if (iputdata["use_creep"]):
            # Center for lining creeping.
            liningCenter = self.get_closest(centeravg, lining)[0]
            lining = self.onlyConnected(lining, liningCenter, iputdata["creep_radius"])
            self.print_this("ver-1", str(len(lining)) + " lining atoms left...")

        # dictionary for covalent radiuses in angstroms
        cov_rad_dict = {
        "ag": 1.53, "as": 1.19, "br": 1.14, "c": 0.77, "ca": 1.74,
        "cd": 1.48, "cl": 0.99, "co": 1.26, "cu": 1.38, "cs": 2.25,
        "f": 0.71, "fe": 1.25, "h": 0.37, "hg": 1.49, "i": 1.33,
        "mg": 1.30, "mn": 1.39, "mo": 1.45, "n": 0.75, "na": 1.54,
        "ni": 1.21, "o": 0.73, "p": 1.04, "pb": 1.47, "pt": 1.28, "s": 1.02,
        "se": 1.16, "v": 1.25, "zn": 1.31
        }

        # The cool stuff. New atoms in optimal positions in relation to charged
        # atoms in the protein.
        # Let's first construct some residues..
        for key, value in iputdata["chains_dict"].items():
            # build internal AA bonds.
            self.build_res(value, cov_rad_dict, iputdata["res_tolerance"], False)
            self.print_this("ver-1", "Residues built for chain " + key)
        # connect residues to one another.
        self.connect_residues(iputdata["AA_dict"], iputdata["protatoms"])
        self.print_this("ver-1", "Residues connected.")
        # And the same for hetatom residues, but this time let's save the
        #residue dictionary.
        hetres = self.build_res(iputdata["hetatoms"], cov_rad_dict, iputdata["res_tolerance"], False)[1]


        self.build_wat(iputdata["water_only"], cov_rad_dict, iputdata["res_tolerance"])
        # Then identify metal containing ligand molecules, and add them to a
        # dedicated list:
        metals = iputdata["metal_coord"]
        metal = []
        hashem = False
        agon = []

        metal_empty_space = set()
        # And then start iterating over all ligand atoms.
        for key, value in iputdata["ligands"].items():
            for atom in value:
                if (atom.elemen.lower() in metals):
                    if (iputdata["agon_model"] and atom.elemen.lower() == "fe"):
                        agon.append(atom)
                    metal.append(atom)
                    metal_empty_space.add(atom.chainid + "-" + str(atom.resseq))
                    # since coordination bonds may come from hetatom residues or
                    # protein atoms, we need to recheck them both.
                    atom.bonds = self.get_coord_bonds(atom, iputdata["protatoms"], iputdata["hetatoms"], iputdata["metal_coord"][atom.elemen.lower()])

        self.print_this("ver-1", str(len(metal)) + " metal atoms found.")
        for met_a in metal:
            if "hem" in met_a.resname.lower().strip():
                hashem = True

        iputdata["lining"] = lining

        if (iputdata["lining_only"]):
            self.print_this("inf", "Outputting lining atoms...")
            return [["lining", lining, None]]

        self.print_this("ver-1", "Creating boundaries.")
        pocketBoundaries = None
        if (iputdata["bounds_only"]):
        # Create boundaries: (basically just a huge mass of "atoms", in form of
        # rays from centeravg to each lining atom.) Better system needed.
        # However, determining whether point (x, y, z) is inside an irregular
        # shape defined by other points appears to be rather hard.
            pocketBoundaries = []
            if iputdata["multibounds"]:
                for c in iputdata["centers"]:
                    pocketBoundaries.extend(self.createBoundaries(lining, iputdata["pocketIDincrement"], 1.5, c))
            else:
                pocketBoundaries = self.createBoundaries(lining, iputdata["pocketIDincrement"], 1.5, centeravg)
            self.print_this("inf", "Outputting boundary atoms...")
            return [["boundaries", pocketBoundaries, None]]

        # Slap a box of atoms on the pocket:
        boxcenters = [centeravg]
        if (not self.is_null(iputdata["box_center"])):
            boxcenters = self.fetch_atoms([iputdata["box_center"]], iputdata["allatoms"], iputdata["water_only"])[0]

        if (iputdata["multibox"]):
            boxcenters = iputdata["centers"]
            far = self.get_farthest_apart_basic(boxcenters)
            newcenters = [] + boxcenters
            increment = iputdata["boxradius"] #iputdata["fillerradius"]*3.5
            for i, b in enumerate(boxcenters):
                for j, b2 in enumerate(boxcenters, start=i):
                    newcenters.extend(self.createray(b, b2, increment, 0.7))
            rad = far[2] + iputdata["boxradius"]
            box = self.create_cube([self.get_avg_point(boxcenters)], rad, iputdata["fillerradius"], iputdata["pack_method"])
            pocket = []
            for b_atom in box:
                add = False
                for c_atom in newcenters:
                    if (self.distance(b_atom, c_atom) < iputdata["boxradius"]):
                        add = True
                        break
                if (add):
                    pocket.append(b_atom)
        else:
            pocket = []
            for b in boxcenters:
                self.print_this("ver-1", "Box centered around " + str(b.x) + " " + str(b.y) + " " + str(b.z))
            box = self.create_cube(boxcenters, iputdata["boxradius"], iputdata["fillerradius"], iputdata["pack_method"])
            pocket = self.delete_farther_than(iputdata["boxradius"], box, boxcenters)[0]
        if (iputdata["cube_only"]):
            self.print_this("inf", "One cube coming up..")
            return [["cube", pocket, None]]
        self.print_this("ver-1", "Pocket box created. Number of atoms: " + str(len(pocket)))
        self.print_this("ver-1", "Shrinking the filling set...")

        # Shrink the filling set:
        if type(iputdata["plane_exclusion"]) is list and len(iputdata["plane_exclusion"]) > 2:
            self.print_this("ver-1", "- Applying plane exclusion algorithm..")
            exclu_cen = centeravg
            if (not self.is_null(iputdata["plane_exclusion_center"])):
                exclu_cen = iputdata["plane_exclusion_center"]
            pocket = self.plane_exclude(exclu_cen, iputdata["plane_exclusion"], pocket)
            self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))

        if (not self.is_null(iputdata["delete_radius"])):
            pocket = self.delete_farther_than(iputdata["delete_radius"], pocket, iputdata["protatoms"])[0]

        if (not self.is_null(iputdata["ligand_distance_restriction"])):
            self.print_this("ver-1", "- Applying ligand distance restriction..")
            datalines = iputdata["ligand_distance_restriction"].lower().split(";")
            for line in datalines:
                data = line.split()
                distance = float(data[1])
                residurr = None
                if(len(data[0]) == 1 and data[0].isalpha()):
                    if data[0] in iputdata["chains_dict"]:
                        residurr = iputdata["chains_dict"][data[0]]
                    elif data[0] in iputdata["hetchains_dict"]:
                        residurr = iputdata["hetchains_dict"][data[0]]
                else:
                    if(data[0].upper() in iputdata["ligands_original_numbers"]):
                        residurr = iputdata["ligands_original_numbers"][data[0].upper()]
                if (residurr is None):
                    self.print_this("war", "Could not locate restriction ligand: " + data[0])
                else:
                    self.print_this("ver-1", "Restriction ligand found: " + residurr[0].chainid + str(residurr[0].resseq) + ". Length of the restriction molecule: " + str(len(residurr)) + ".")

                    pocket = self.delete_farther_than(distance, pocket, residurr)[0]
            self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))

        self.print_this("ver-1", "- Considering protein atoms...")

        empty_this_space = []
        for key in list(iputdata["ligands_original_numbers"].keys()):
            if key in metal_empty_space:
                empty_this_space.extend(iputdata["ligands_original_numbers"][key])
        pocket = self.nonOverlapping(empty_this_space, pocket)[0]
        pocket = self.nonOverlapping(iputdata["protatoms"], pocket)[0]
        self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))
        self.print_this("ver-1", "- Checking inclusion...")
        keep = None
        if iputdata["keep_anyway_radius"] > 0.1:
            keep = [iputdata["keep_anyway_radius"], iputdata["keep_anyway_at_least_AA"], iputdata["protatoms"], metal, iputdata["keep_anyway_AA-Resseq-dist"]]

        if (pocketBoundaries is None):
            pocketBoundaries = []
            if iputdata["multibounds"]:
                for c in iputdata["centers"]:
                    pocketBoundaries.extend(self.createBoundaries(lining, iputdata["pocketIDincrement"], 1.5, c))
            else:
                pocketBoundaries = self.createBoundaries(lining, iputdata["pocketIDincrement"], 1.5, centeravg)
        pocket = self.overlappingOnly(pocket, pocketBoundaries, iputdata["protatoms"], keep_anyway=keep)[0]

        for metal_atom in metal:
            pocket = self.nonOverlapping(hetres[metal_atom.resseq], pocket)[0]
        self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))

        if len(pocket) < 1:
            self.print_this("fat-err", "Pocket shrunk to nothing. \n exiting...")
            self.quit_now()
        center = self.newcenter(centeravg, pocket, iputdata["protatoms"])
        # Blast away some atoms with angle information if necessary
        if (not self.is_null(iputdata["angledata"])):
            self.print_this("ver-1", "- Angle blasting...")
            pocket = self.angleBlast(pocket, iputdata["allatoms"], center, iputdata["angledata"])
            center = self.newcenter(center, pocket, iputdata["protatoms"])
            self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))

        atom_to_atom = iputdata["fillerradius"]*2
        first_hyp = math.sqrt((atom_to_atom*atom_to_atom) + (iputdata["fillerradius"]*iputdata["fillerradius"]))
        bond_length =  0.001 + math.sqrt((first_hyp * first_hyp) + (iputdata["fillerradius"]*iputdata["fillerradius"]))#atom_to_atom + 0.1 # #math.sqrt(((atom_to_atom)*(atom_to_atom)) + ((atom_to_atom)*(atom_to_atom)))
        # Only get connected atoms(eliminate secondary pockets, if any)
        if (not iputdata["keepsecondary"]):

            # Add bonds:
            bonds = self.add_bonds(pocket, bond_length)
            #~ for a in pocket:
                #~ a.resname = "TAS"
                #~ a.resseq = 1
            remov = iputdata["del_angles"]
            max_number = iputdata["del_min_size"]
            self.print_this("ver-1", "- Removing unconnected pocket atoms...")
            if (iputdata["pack_method"] in ["bcc", "fcc"]):
                # removes all atoms with only two bonds given angles and more
                # than max_number of atoms on one side
                pocket = self.remove_unconnected(self.get_closest(center, pocket)[0], pocket, bonds, remov, max_number)
            else:
                pocket = self.remove_gaps(self.get_closest(center, pocket)[0], pocket, atom_to_atom + 0.15)

            self.print_this("ver-1", "- - Pocket size: " + str(len(pocket)))

            bonds = self.add_bonds(pocket, bond_length)

        lining = self.get_within_radius_of_list(6, lining, pocket)

        if (len(iputdata["nofillaage"]) > 0):
            durpader = {}
            for a in iputdata["nofillaage"]:
                if a.resname in durpader:
                    durpader[a.resname].append(a)
                else:
                    durpader[a.resname] = [a]
            new_fillage = []
            for key in list(durpader.keys()):
                new_fillage.append(durpader[key])
            iputdata["nofillaage"] = new_fillage
        check_for_overlap = empty_this_space + iputdata["protatoms"]
        do_wats = False
        if (iputdata["h_bond_limit"] >= 0):
            do_wats = True
        d_opt = None
        if (iputdata["basic_ch_atoms"]):
            charge_wat = [None, None, None]
            if (do_wats):
                # dictionary for amino acid angles.
                # TODO: input
                angles_dict = {
                        "main_chain": {"n": [120, 120], "o": [120, 120]},
                        "thr": {"o": [109.5, 109.5]},
                        "ser": {"o": [109.5, 109.5]},
                        "tyr": {"o": [109.5, 109.5]},
                        "trp": {"n": [120, 120]},
                        "asp": {"o": [120, 109.5]},
                        "glu": {"o": [120, 109.5]},
                        "asn": {"n": [120, 120], "o": [120, 120]},
                        "gln": {"n": [120, 120], "o": [120, 120]},
                        "lys": {"n": [109.5, 109.5]},
                        "his": {"n": [120, 120]},
                        "hie": {"n": [120, 120]},
                        "hid": {"n": [120, 120]},
                        "hip": {"n": [120, 120]},
                        "arg": {"n": [120, 120]}
                        }
                self.print_this("ver-1", "Calculating water..")
                waters_four = self.check_waters(iputdata["water_only"], lining, iputdata["hetatoms"], angles_dict, iputdata["charged_rad"], iputdata["fillerradius"], iputdata["H_bond_distance"], iputdata["H_add"], bond_no=iputdata["h_bond_limit"], dist=iputdata["hobomax"])


            c = self.place_optimals(lining, metal, pocket, iputdata["fillerradius"], iputdata["protatoms"], charge_wat[1], iputdata["wat_conv_distance"], iputdata["charged_rad"], iputdata["H_bond_distance"], iputdata["H_add"], iputdata["charge_lib"], angle_tolerance=iputdata["angle_tolerance"])
            circles = c[0]
            options = c[1]
            no_recheck = c[2]

            # Find metal coordination atoms and add the ones not for heme.
            # Those come later. Don't add if the atom overlaps with "nofillaage"
            # or metal-containing residue.
            optimal_metals_heme = []
            if(len(metal) > 0):
                # Check residues for which overlap is not wanted, and divide
                # metals to heme and non-heme groups.
                keep_resname = set()
                metals_not_heme = []
                metals_heme = []
                for a in metal:
                    keep_resname.add(a.resname)
                    if a.resname.lower() == "hem":
                        metals_heme.append(a)
                    else:
                        metals_not_heme.append(a)
                hetcheck = []
                for entry in iputdata["nofillaage"]:
                    hetcheck.extend(entry)
                for heta in iputdata["hetatoms"]:
                    if heta.resname in keep_resname:
                        hetcheck.append(heta)
                checkset = self.get_within_radius_of_set(5, [iputdata["protatoms"], hetcheck], metal)
                # Find metal coordination atoms.
                optimal_metals_not_heme = self.place_optimal_metals(metals_not_heme, iputdata["metal_coord"], iputdata["fillerradius"], checkset, iputdata["charged_rad"], pocket, iputdata["H_bond_distance"])
                circles.extend(optimal_metals_not_heme)
                optimal_metals_heme = self.place_optimal_metals(metals_heme, iputdata["metal_coord"], iputdata["fillerradius"], checkset, iputdata["charged_rad"], pocket, iputdata["agon_dist"])

            d = []

            for c in circles:
                for a in c:
                    d.append(a)

            if do_wats:
                water_temp = self.nonOverlapping(d, waters_four)[0]
                d.extend(water_temp)

            # Add coordination points for "nofillaage" residues except water
            if (len(iputdata["nofillaage"]) > 0):
                for entry in iputdata["nofillaage"]:
                    if (len(entry) > 1) and entry[0] not in iputdata["water_only"]:
                        self.build_single_res(entry, cov_rad_dict, iputdata["res_tolerance"])
                        nof_opt = self.optimal_for_res(entry, iputdata["charged_rad"], iputdata["cofactor_coord"], iputdata["H_bond_distance"], iputdata["H_add"], iputdata["protatoms"])
                        if len(nof_opt[0]) > 0:
                            d = self.nonOverlapping(nof_opt[0], d)[0] ## ADDED 16-2-2014
                            d.extend(nof_opt[0])
                        if len(nof_opt[1]) > 0:
                            options.update(nof_opt[1])

            self.print_this("ver-1", str(len(d) + sum(len(v) for v in options.values())) + " atoms placed near charged groups.")
            if (iputdata["ch_only"]):
                self.print_this("inf", "Kicking out all charged atoms..")
                optd = []
                aidx = 1
                for resid, mods in options.items():
                    for m in mods:
                        optd.extend(m)
                        for atom in m:
                            atom.idn = aidx
                            atom.name = atom.name+str(aidx)
                            if(atom.host is not None):
                                atom.resname = atom.host.resname
                                atom.resseq = atom.host.resseq
                            else:
                                atom.resname = "UKN"
                                atom.resseq = 9999
                            if atom.tempf < 0:
                                atom.elemen = "O"
                            else:
                                atom.elemen = "N"
                            aidx += 1
                for atom in d:
                    atom.idn = aidx
                    if(atom.host is not None):
                        atom.resname = atom.host.resname
                        atom.resseq = atom.host.resseq
                    else:
                        atom.resname = "UKN"
                        atom.resseq = 9999
                    if atom.tempf < 0:
                        atom.elemen = "O"
                    else:
                        atom.elemen = "N"
                    atom.name = atom.name+str(aidx)
                    aidx += 1
                ret =   [
                        ["CH_only", d, None],
                        ["Optional_mods", optd, None]
                        ]
                return ret

            # Delete atoms which overlap with or are too far from protein.
            self.print_this("ver-1", "Applying restrictions..")
            d_opt = None

            d = self.nonOverlapping(iputdata["protatoms"], d)#, cir_cheat)

            do_over = d[1]
            d = d[0]
            no_recheck=set()
            # Only get the atoms that are close enough to the pocket atoms.
            #TODO: radcheat inputtiin
            d = self.delete_farther_than(iputdata["optimal_inclusion"], d, pocket) #s.overlappingOnly(d, pocket, iputdata["protatoms"], radcheat = new_radcheat )
            do_over.extend(d[1])
            d = d[0]
            # Check deleted atoms again. perhaps a slight change in angle or
            # distance can fix it?
            recheck = self.check_again(do_over, lining, pocket, iputdata["angle_tolerance"], iputdata["fillerradius"], no_recheck, iputdata["charged_rad"], iputdata["H_bond_distance"], iputdata["H_add"])
            d.extend(recheck)

        for entry in iputdata["nofillaage"]:
            check_for_overlap.extend(entry)
        d = self.delete_farther_than(iputdata["optimal_inclusion"], d, pocket)[0]
        d = self.nonOverlapping(check_for_overlap, d)[0]
        # Remove overlapping non-charged atoms
        pocket = self.nonOverlapping(d, pocket)[0]
        self.set_ch(pocket, iputdata["protatoms"], iputdata["charge_radius"])

        if (not self.is_null(iputdata["exclusion_zone"][0])):
            AAs = set()
            for num in iputdata["exclusion_zone"]:
                AAs.add(int(num))
            pocket = self.within_bounds(pocket, self.get_AAs(AAs, iputdata["protatoms"]))[1]
        ret = []
        bonds = None
        if (iputdata["debug"]):
            bonds = []
            bonds = self.add_bonds(pocket, bond_length)

        # Create agonist and antagonist models if relevant.
        # - Check 1) coordination atom distance from pocket and 2) overlaps
        # - Create also agonist coordination atom with opposite charge
        # - Build
        met_addition = []
        if (len(optimal_metals_heme) > 0):
            for c in optimal_metals_heme:
                for a in c:
                    met_addition.append(a)
            met_addition = self.delete_farther_than(iputdata["optimal_inclusion"], met_addition, pocket)[0] #s.overlappingOnly(met_addition, pocket, iputdata["protatoms"], radcheat = new_radcheat )[0]
            pocket = self.nonOverlapping(met_addition, pocket)[0]
            d = self.nonOverlapping(met_addition, d)[0]
        defname = iputdata["outputname"]
        defpocket = pocket + d #d_wat + d
        if (len(met_addition) < 1):
            iputdata["agon_model"] = False
        if (iputdata["agon_model"] and iputdata["dual_model"] and hashem):
            ret.append([defname + "-antagonist", defpocket + met_addition, bonds])
        if (iputdata["agon_model"] and hashem):
            # Find closest atom on a circle to the "NA" atom in heme molecule
            # It is supposed that there is only one heme in the protein
            # structure (agon[0])
            heme_na = None
            for contact_atom in agon[0].bonds:
                if contact_atom.name.lower() == "na":
                    heme_na = contact_atom
            closest_atom = None
            closest_distance = 10000.0
            agonist_points = self.get_circle(met_addition[0], agon[0], 120, iputdata["agon_dist"], 20, 0.0)
            for atom in agonist_points:
                if self.distance(atom, heme_na) < closest_distance:
                    closest_atom = atom
                    closest_distance = self.distance(atom, heme_na)
            # Set charge and radius for the closest atom and add it on the
            # pocket model
            closest_atom.tempf = -met_addition[0].tempf
            closest_atom.radius = met_addition[0].radius
            agonist_pocket = [copy(atom) for atom in pocket] + [copy(atom) for atom in d]
            agonist_pocket = self.nonOverlapping([closest_atom], agonist_pocket)[0] + [closest_atom]
            ret.append([defname + "-agonist", agonist_pocket, bonds])
        else:
            ret.append([defname,  defpocket, bonds])

        self.print_this("ver-1", "Finishing up..")
        if (d_opt is not None):
            options.update(d_opt)
        # Create alternative charge models
        if not iputdata["old_dupl"]: # (self.not_empty(options) and not iputdata["basic_ch_atoms
            mod_sets = []
            new_opt = []
            appendage = []
            for key in list(options.keys()):
                newmods = []
                for mod in options[key]:
                    # Alternative model: remove atoms which overlap with.. what?
                    mod_fin = self.nonOverlapping(check_for_overlap+empty_this_space, mod)[0]
                    # Alternative model: remove atoms which are too far from
                    # neutral atoms
                    mod_fin = self.delete_farther_than(iputdata["optimal_inclusion"], mod_fin, pocket)[0]
                    #~ if key == "155":
                        #~ print(mod[0].host.name)
                        #~ print(mod_fin)
                        #~ print("...")
                    #newmods.append(mod)
                    if len(mod_fin) > 0:
                        newmods.append(mod_fin)
                if len(newmods) > 1:
                    new_opt.append(newmods)
                elif len(newmods) > 0:
                    appendage.extend(newmods)
            options = new_opt
            i = 0
            for item in product(*options):
                if len(item) > 0:
                    temp = []
                    for l in list(item):
                        temp.extend(l)
                    mod_sets.append(temp)

            new_sets = []
            for i, temp_mod in enumerate(mod_sets):
                nup = temp_mod
                if len(nup) > 0:
                    temp =  self.not_included(nup, new_sets)
                    if (temp[0]):
                        new_sets.append(temp[1])
                    elif temp[2] > -1:
                        new_sets[temp[2]] = nup
            final_mods = []
            ap_mods = []
            for ap in appendage:
                ap_mods.extend(ap)
            if len(new_sets) > 0:
                for n in new_sets:
                    for a in [copy(atom) for atom in ret]:
                        newmod = [a[0], self.fix_overlap([copy(atom) for atom in n] + [copy(atom) for atom in a[1]] + [copy(atom) for atom in ap_mods]), a[2]] #s.nonOverlapping(check_for_overlap, newmod)[0], a[2]]
                        final_mods.append(newmod)
                final_mods.sort(key=lambda x: x[0])
                ret = final_mods

        elif (len(options) > 0 and iputdata["basic_ch_atoms"]):# and not iputdata["no_options"]):
            surviving_mods = {}
            print_info = False
            optstring = "No one model could be created for AAs: \n"
            for key in list(options.keys()):
                surv_number = 0
                for model in options[key]:
                    do_over = []
                    model = self.nonOverlapping(empty_this_space, model)[0]
                    mod = self.nonOverlapping(iputdata["protatoms"], model)#, cir_cheat)
                    do_over.extend(mod[1])
                    mod =  mod[0]
                    mod = self.delete_farther_than(iputdata["optimal_inclusion"], mod, pocket) #s.overlappingOnly(mod, pocket, iputdata["protatoms"], radcheat = new_radcheat)
                    do_over.extend(mod[1])
                    mod = mod[0]
                    redone = self.check_again(do_over, lining, pocket, iputdata["angle_tolerance"], iputdata["fillerradius"], no_recheck, iputdata["charged_rad"], iputdata["H_bond_distance"], iputdata["H_add"])
                    mod.extend(redone)

                    if (len(mod) > 0):
                        surv_number += 1
                        if key not in surviving_mods:
                            surviving_mods[key] = []
                        surviving_mods[key] += [mod]
                if surv_number > 0: #and not key == d_name:
                    optstring += key + ", "
                    print_info = True
            optstring = optstring[:-2] + "\n"

            combination_heap = []
            combinations = []
            for key in list(surviving_mods.keys()):
                combination_heap.append(surviving_mods[key])

            if len(combination_heap) > 0:
                combus = []
                for item in product(*combination_heap):
                    temp = []
                    for l in list(item):
                        temp.extend(l)
                    combus.append(temp)

                newret = []
                testset = []
                idx = 1
                for co in combus:
                    for r in ret:
                        r[1] = self.nonOverlapping(co, r[1])[0]
                        new_un = r[1] + co

                        temp = self.included(new_un, testset)
                        if (temp[0]):
                            if (temp[1] > new_un):
                                testset[temp[2]] = temp[1]
                                newret[temp[2]] = [r[0] + "-" + str(idx), temp[1], r[2]]
                            else:
                                testset[temp[2]] = new_un
                                newret[temp[2]] = [r[0] + "-" + str(idx), new_un, r[2]]
                        else:
                            testset.append(new_un)
                            newret.append([r[0] + "-" + str(idx), new_un, r[2]])

                    idx += 1
                newret.sort(key=lambda x: x[0].split("-")[1])
                ret = newret

                if (print_info):
                    optstring += "Thus, optional models with various combinations have been APPENDED to the same FILE."
        sizes = []
        ret = self.number_and_name_these(ret, "TAS", iputdata["ignore_charge_limit"])
        for idx, l in enumerate(ret):
            model = l[1]
            for entr in iputdata["nofillaage"]:
                model = self.nonOverlapping(entr, model)[0]
            l[1] = model
            sizes.append(len(model))

        if (len(ret) > 1):
            prt = "Final pocket sizes: " + str(sizes) + "\nNumber of models: " + str(len(ret))
        else:
            prt = "Final pocket size: " + str(sizes)
        self.print_this("inf", prt)
        return ret

    def get_add_lines(self, metal_coordinate_atoms, lining_atoms, h_dist, h_add):
        oh_group = set(["ser", "thr", "tyr"])
        planar_o = set(["asp", "glu", "asn", "gln"])
        planar_n = set(["arg", "gln", "asn"])
        tetra_n = set(["lys"])
        ring_n = set(["his", "hip", "hid", "hie", "trp"])
        acceptor_only = set(["asn", "gln"])
        double_o_and_oh = set([])#"ash", "glh", "asp", "glu"])

        #cycl_c = set(["his", "tyr", "phe", "trp"])

        del_set = []
        done_set = set()
        addery = []

        for atom in lining_atoms:
            if atom not in metal_coordinate_atoms and self.is_charged(atom): # and not atom in done_set):
                #cont = True
                if (atom.elemen.lower() == "h"):
                    add = False
                    ref = atom.bonds[0]
                    for a in atom.bonds:
                        if a.elemen.lower() == "n" or a.elemen.lower() == "o":
                            ref = a
                            if a.name.lower() != "n" and a.name.lower() != "o":
                                if a.resname.lower() in tetra_n or a.resname.lower() in oh_group or a.resname.lower() in double_o_and_oh:
                                    add = True

                        elif a.elemen.lower() == "h":
                            self.print_this("err", "H-H bond detected? Weird.")
                            self.print_this("err", "Atoms: " + str(atom.idn) + " and " + str(a.idn))
                    if add:
                        addery.append([atom, ref, 180, "acceptor", "linear-h"])
                        done_set.add(ref)
                elif(atom.name.lower() == "o"):
                    addery.append([atom, atom.bonds[0], 120, "acceptor", "planar"])
                # Main chain nitrogen
                elif (atom.name.lower() == "n" and not atom.resname.lower() == "pro"):
                    refs = []
                    h_at = None
                    for ba in atom.bonds:
                        if ba.elemen.lower() == "c":
                            refs.append(ba)
                        elif ba.elemen.lower() == "h":
                            h_at = ba
                    if h_at is not None:
                        ref = self.get_to_from(atom, h_at, self.distance(h_at, atom) + h_dist, 0.5)
                    else:
                        if len(refs) > 2:
                            newref = []
                            for rat in refs:
                                add = True
                                for nat in newref:
                                    if (self.distance(rat, nat) < 0.1):
                                        add = False
                                if (add):
                                    newref.append(rat)
                            refs = newref
                        ref = self.get_avg_point(refs)
                    addery.append([atom, ref, 180, "donor", "linear"])
                # Side chain nitrogens
                elif(atom.elemen.lower() == "n" and not atom.resname.lower() == "pro"):
                    if (atom.resname.lower() in planar_n ):
                        distance = "donor"
                        c_bonds = []
                        h_bonds = []
                        for ba in atom.bonds:
                            if ba.elemen.lower() == "c":
                                c_bonds.append(ba)
                            elif ba.elemen.lower() == "h":
                                h_bonds.append(ba)

                        ref = c_bonds[0]
                        angle = 120
                        conf = "planar"
                        if (atom.resname.lower() == "arg"):
                            if (len(h_bonds) == 2):
                                for h_at in h_bonds:
                                    addery.append([h_at, atom, 180, "acceptor", "linear"])
                            elif (len(h_bonds) == 1 and len(c_bonds) == 2):
                                addery.append([h_bonds[0], atom, 180, "acceptor", "linear"])
                            elif len(c_bonds) > 1:
                                addery.append([atom, self.get_avg_point(c_bonds), 180, "donor", "linear"])
                            elif len(c_bonds) > 0:
                                addery.append([atom, c_bonds[0], 120, "donor", "planar"])
                            else:
                                self.print_this("err", atom.resname + str(atom.resseq) + "-" + atom.name + " seems to have no bound carbons. Interesting. Ignoring...")
                        else:
                            addery.append([atom, ref, angle, distance, conf])
                    elif (atom.resname.lower() in ring_n):
                        distance = "donor"
                        if (atom.resname.lower() in ring_n and atom.resname.lower() != "trp"):
                            if (atom.resname.lower() == "his"):
                                h_num = 0
                                for a in atom.myAA:
                                    if a.name.lower() == "ne":
                                        for ba in a.bonds:
                                            if ba.elemen.lower() == "h":
                                                h_num += 1
                                                atom.resname == "hie"
                                    if a.name.lower() == "nd":
                                        for ba in a.bonds:
                                            if ba.elemen.lower() == "h":
                                                h_num += 1
                                                atom.resname == "hid"
                                if h_num > 1:
                                    atom.resname == "hip"
                        if ((atom.resname.lower() == "hid" and "ne" in atom.name.lower()) or (atom.resname.lower() == "hie" and "nd" in atom.name.lower()) or (atom.resname.lower() == "his" and "ne" in atom.name.lower())):
                            distance = "acceptor"
                        refs = []
                        for ba in atom.bonds:
                            if ba.elemen.lower() == "c":
                                refs.append(ba)
                        addery.append([atom, self.get_avg_point(refs), 180, distance, "linear"])
                    elif (atom.resname.lower() in tetra_n):
                        ref_a = None
                        for a in atom.bonds:
                            if a.elemen.lower() == "c":
                                ref_a = a
                        if ref_a:
                            addery.append([atom, a, 109.5, "donor", "tetrahedral"])
                elif(atom.elemen.lower() == "o"):

                    if (atom.resname.lower() in planar_o):
                        addery.append([atom, atom.bonds[0], 120, "acceptor", "planar"])
                    elif(atom.resname.lower() in double_o_and_oh):
                        #if( len(atom.bonds) < 2):
                        cref = atom.bonds[0]
                        href = None

                        for a in atom.bonds:
                            if a.elemen.lower() == "c":
                                cref = a
                            elif a.elemen.lower() == "h":
                                href = a

                        app = [atom, a, 120, "acceptor", "planar"]
                        if href is not None:
                            app[2] = 109.5
                        #if (atom.am_oh):
                        #    app[2] = 109.5
                        addery.append(app)
                        #else:
                            #addery.append([atom, atom.bonds[0], 109.5, "2acceptor-1donor", "tetrahedral"])
                    elif (atom.resname.lower() in oh_group):
                        addery.append([atom, atom.bonds[0], 109.5, "2acceptor-1donor", "tetrahedral"])

        return [addery, del_set, done_set]


    # Look for hydrogen that should be connected to atom
    # from donor_groups. Return opposite charge of the
    # hydrogen (=acceptor charge) or None if hydrogen was
    # not found.
    def find_acceptor_charge(self, atom, charge_lib):
        if atom.name.lower() == "n": # Main chain
            return -charge_lib[atom.resname.lower()]["h"]
        elif atom.name.lower() in self.donor_groups:
            for hydrogen in self.donor_groups[atom.name.lower()]:
                if hydrogen in charge_lib[atom.resname.lower()]:
                    return -charge_lib[atom.resname.lower()][hydrogen]
        return None


    def do_lines(self, pocket, addery, protein_atoms, del_set, acceptor, H_add, radius, check_water, angle_tolerance, done_set, chargerad, charge_lib):
        circles = []
        optionals = {}
        aspglu_done = set()
        aspglu_undone = set()
        same_ch = []
        no_recheck = set()

        for entry in addery:
            atom = entry[0]
            if atom in done_set:
                continue
            ref = entry[1]
            angle = entry[2]
            #mode = entry[3]
            configuration = entry[4]
            distance = acceptor
            ch = -atom.new_ch
            ch_atom = atom

            #ag_refpoint = None
            aspglu_ref = None
            aspglu_don = False
            optionals_placed = False
            atm = []

            if(entry[3] == "donor"):

                distance += H_add
            hyd_ch_atom = None

            if (atom.elemen.lower() == "h" and configuration == "linear-h"):
                main_atm = ref
                ch = -main_atm.new_ch
                ch_atom = main_atm
                hyd_ch_atom = main_atm
                c_atom = None
                for a in main_atm.bonds:
                    if a.elemen.lower() == "c":
                        c_atom = a
                if c_atom is None:
                    self.print_this("err", "No carbon atom found for " + main_atm.resname + str(main_atm.resseq) + "-" + main_atm.name)
                else:
                    distance = acceptor
                    h_bond_atm = self.get_to_from(atom, main_atm, acceptor + self.distance(main_atm, atom), radius)
                    if (main_atm.elemen.lower() == "n"):
                        distance += self.distance(main_atm, atom)
                    cir = self.get_circle(main_atm, c_atom, 109.5, distance, 5, radius)
                    closest_to_h = self.get_closest(h_bond_atm, cir)
                    start_idx = closest_to_h[2]
                    closest_to_h = closest_to_h[0]
                    increment = len(cir)/3

                    if "h" not in [x.elemen.lower() for x in main_atm.bonds if x != atom]:
                        two_others = [cir[int(start_idx - increment)], cir[int(start_idx - (2*increment))]]
                        atm.extend(two_others)
                    atm.append(h_bond_atm)

            elif (angle == 180):
                aff = self.get_circle(atom, ref, angle, distance, 10, radius)
                atm.extend(aff)
            elif (angle == 120):
                append = True
                circle = self.get_circle(atom, ref, angle, distance, 5, radius)
                dun = []
                ref2 = None
                for bonded in atom.bonds:
                    if bonded.elemen.lower() != "h":
                        for bonded_secondary in bonded.bonds:
                            if bonded_secondary.elemen.lower() != "h" and bonded_secondary != atom:
                                ref2 = bonded_secondary
                if not ref2:
                    self.print_this("err", "Not enough reference atoms found for {}{}-{}. Skipping coordination atom addition.".format(atom.resname, str(atom.resseq), atom.name))
                    continue
                candidates = []
                for c_a in circle:
                    arr = [self.getvec(c_a, atom), self.getvec(ref, atom), self.getvec(ref2, atom)] #[c_a.x - atom.x, c_a.y - atom.y, c_a.z - atom.z], [ref.x - atom.x, ref.y - atom.y, ref.z - atom.z], [ref2.x - atom.x, ref2.y - atom.y, ref2.z - atom.z]]
                    det = self.array_det(arr)
                    candidates.append([c_a, math.fabs(det)])
                candidates.sort(key=lambda x: x[1])
                for a in candidates[:2]:
                    dun.append(a[0])

                if False: #(atom.resname.lower() == "asp" or atom.resname.lower() == "glu"):
                    if atom.resseq not in aspglu_done:
                        for p in protein_atoms:
                            if (self.distance(p, atom) < 3.4 and atom.resseq != p.resseq and ref.idn != p.idn ):
                                if (self.is_charged(p)):
                                    vecp = self.getvec(p, atom)
                                    refvec = self.getvec(ref, atom)
                                    ang = self.angle(vecp, refvec)
                                    if (angle + angle_tolerance) > ang > (angle - angle_tolerance):
                                        if ((109.5 - ang) > (120-ang)) or atom.resseq in aspglu_undone:
                                            aspglu_ref = p
                                            angle = 109.5
                                            aspglu_done.add(atom.resseq)
                                            if (p.elemen.lower() == "h" or p.elemen.lower() == "n"):
                                                if not (p.name.lower() == "nd" and p.resname.lower() == "hie"):
                                                    aspglu_don = True
                                                    break
                                                else:
                                                    aspglu_don = False
                                            else:
                                                aspglu_don = False
                                            append = False
                                        else:
                                            aspglu_undone.add(atom.resseq)
                if (append):
                    atm.extend(dun)
            if (angle == 109.5 and configuration == "tetrahedral"):
                if (atom.elemen.lower() == "n" and len(atom.bonds) > 4):
                    self.print_this("war", atom.elemen.upper() + "H group with more than four bonded atoms found! \nResid: " + str(atom.resseq) + "\nIGNORING ATOM")
                elif (len(atom.bonds) < 1):
                    self.print_this("war", atom.elemen.upper() + " atom with 0 bonded atoms found! \nResid: " + str(atom.resseq) + "\nIgnoring...")
                    continue

                if (atom.elemen.lower() == "o" ):
                    c_atom = None
                    others = []
                    h_at = []
                    for a in atom.bonds:
                        if a.elemen.lower() == "c":
                            c_atom = a
                        elif a.elemen.lower() == "h":
                            h_at.append(a)
                        else: others.append(a)

                    wat_ref = None
                    don_ref = None
                    acc_ref = None
                    if len(others) > 0:
                        for a in others:
                            if self.is_water(a):
                                wat_ref = a
                                if (atom in a.acceptor_bonds):
                                    acc_ref = a
                                if (atom in a.donor_bonds):
                                    don_ref = a
                    if (len(h_at) > 0):
                        don_ref = h_at[0]

                    ref = c_atom


                    onemod = False
                    twomods = False
                    #threemods = True
                    refh = None
                    if don_ref is not None:
                        onemod = True
                        refh = don_ref
                    elif acc_ref is not None:
                        twomods = True
                        refh = acc_ref
                    else:
                        #threemods = True
                        refh = wat_ref
                    if refh is None:
                        refh = self.get_avg_point(pocket)

                    if (aspglu_ref is not None):
                        refh = aspglu_ref
                        if (aspglu_don):
                            onemod = True

                    dist = 1.6

                    circle1 = self.get_circle(atom, ref, 109.5, dist, 5, 0.7)
                    circle2 = self.get_circle(atom, ref, 109.5, dist+H_add, 5, 0.7)

                    incr = int(len(circle1) / 3)

                    mods = []
                    closest1 = self.get_closest(refh, circle1)
                    closest2 = self.get_closest(refh, circle2)
                    if (onemod):
                        mods = [[closest2[0], circle1[closest1[2]-incr], circle1[closest1[2] - (2*incr)]]]
                        no_recheck.add(atom)
                    elif (twomods):
                        first = [closest1[0], circle2[closest2[2]-incr], circle1[closest1[2] - (2*incr)]]
                        second = [closest1[0], circle1[closest1[2]-incr], circle2[closest2[2] - (2*incr)]]
                        mods = [first, second]
                    else:
                        first = [closest2[0], circle1[closest1[2]-incr], circle1[closest1[2]-(2*incr)]]
                        second = [circle2[closest2[2]-incr], closest1[0], circle1[closest1[2]-2*incr]]
                        third = [circle2[closest2[2]-(2*incr)], circle1[closest1[2]-incr], closest1[0]]
                        mods = [first, second, third]

                    finmods = []#[circle1, circle2]

                    for opt in mods:
                        opt = self.not_in_the_way(opt, check_water, atom)
                        finmods.append(opt)
                    refh.elemen = "MG"
                    refh.name = "MG"


                    if len(mods) == 1:
                        atm = finmods[0]
                        atm.append(refh)
                    else:
                        if not str(atom.resseq) in optionals:
                            optionals[str(atom.resseq)] = finmods
                        else:
                            optionals[str(atom.resseq)].extend(finmods)
                        optionals_placed = True

                elif (atom.elemen.lower() == "n"):# or atom.elemen.lower() == "c"):
                    dun = []
                    if (len(atom.bonds) > 1):
                        refatoms = []
                        for b in atom.bonds:
                            if (b.elemen.lower() == "h"):
                                refatoms.append(b)
                        if (len(refatoms) < 3):
                            circle = self.get_circle(atom, ref, angle, distance + H_add, 5, radius)
                            if len(refatoms) < 1:
                                refatoms.append(self.get_closest(self.get_avg_point(pocket), circle)[0])
                            ref_temp = self.get_closest(refatoms[0], circle)
                            dun.append(ref_temp[0])
                            dun.append(circle[ref_temp[2] - int(len(circle)/2)])
                            dun.append(circle[ref_temp[2] - int((len(circle)/2)*2)])
                        if (len(refatoms) == 3):
                            for a in refatoms:
                                dun.append(self.get_to_from(a, atom, distance, 0.7))

                        atm.extend(dun)
                    elif (len(atom.bonds) == 1):
                        circle = self.get_circle(atom, ref, angle, distance, 5, radius)
                        first = self.get_closest(self.get_avg_point(pocket), circle)
                        f_atom = first[0]
                        f_idx = first[2]

                        second_atom = circle[int(f_idx - (len(circle)/3))]
                        third_atom = circle[int(f_idx - ((len(circle)/3)*2))]
                        dun.extend([f_atom, second_atom, third_atom])

                        opt_idx = int(f_idx - ((len(circle)/6)*1))
                        opt_first_atom = circle[opt_idx]
                        opt_second_atom = circle[int(opt_idx - (len(circle)/3))]
                        opt_third_atom = circle[int(opt_idx - ((len(circle)/3)*2))]

                        en = [opt_first_atom, opt_second_atom, opt_third_atom]

                        tempo = [dun, en]
                        for opt in tempo:
                            opt = self.not_in_the_way(opt, check_water, atom)

                        if not str(atom.resseq) in optionals:
                            optionals[str(atom.resseq)] = tempo
                        else:
                            optionals[str(atom.resseq)].extend(tempo)
                        optionals_placed = True
            atm = self.merge_close(atm, 0.2)
            for a in atm:
                h_cand = None
                for bonded_atom in ch_atom.bonds:
                    if bonded_atom.elemen.lower() == "h":
                        mydist = self.distance(bonded_atom, a)
                        if h_cand == None:
                            h_cand = [bonded_atom, mydist]
                        elif mydist < h_cand[1]:
                            h_cand = [bonded_atom, mydist]
                if h_cand is not None:
                    if self.distance(a, ch_atom) > self.distance(a, h_cand[0]):
                        hyd_ch_atom = h_cand[0]
                if hyd_ch_atom is not None:
                    a.tempf = -hyd_ch_atom.new_ch
                elif self.distance(a, ch_atom) < acceptor + 0.5*H_add:
                    a.tempf = ch
                else:
                    acceptor_charge = self.find_acceptor_charge(atom, charge_lib)
                    if acceptor_charge is None:
                        self.print_this("war", "No hydrogen found for {0}{1}-{2}, using the charge of {2} for coordination atom.".format(atom.resname, atom.resseq, atom.name))
                        acceptor_charge = -ch
                    a.tempf = acceptor_charge
                a.resseq = atom.resseq
                a.host = atom
                a.addline = entry
                a.radius = chargerad

            circles.append(atm)
            if (str(atom.resseq) in optionals and optionals_placed):
                for i, model in enumerate(optionals[str(atom.resseq)]):
                    model = self.merge_close(model, 0.2)
                    for ab in model:
                        h_cand = None
                        for bonded_atom in ch_atom.bonds:
                            if bonded_atom.elemen.lower() == "h":
                                mydist = self.distance(bonded_atom, a)
                                if h_cand == None:
                                    h_cand = [bonded_atom, mydist]
                                elif mydist < h_cand[1]:
                                    h_cand = [bonded_atom, mydist]
                        if h_cand is not None:
                            if self.distance(a, ch_atom) > self.distance(a, h_cand[0]):
                                hyd_ch_atom = h_cand[0]
                        if hyd_ch_atom is not None:
                            a.tempf = -hyd_ch_atom.new_ch
                        elif self.distance(ab, ch_atom) < acceptor + 0.5*H_add:
                            ab.tempf = ch
                        else:
                            acceptor_charge = self.find_acceptor_charge(atom, charge_lib)
                            if acceptor_charge is None:
                                self.print_this("war", "No hydrogen found for {0}{1}-{2}, using the charge of {2} for coordination atom.".format(atom.resname, atom.resseq, atom.name))
                                acceptor_charge = -ch
                            ab.tempf = acceptor_charge
                        ab.radius = chargerad
                        ab.resname = "OPT"
                        ab.resseq = atom.resseq
                        ab.host = atom
                        ab.addline = entry
                    optionals[str(atom.resseq)][i] = model
            for a in same_ch:
                a.tempf = atom.new_ch
        return [circles, optionals, no_recheck]

    # returns atoms from opt that are not in the way of possible hydrogen bonds
    # main_atom might have with waters from check_water.
    def not_in_the_way(self, opt, check_water, main_atom):
        if (check_water is None):
            return opt
        ret = []
        for o in opt:
            safe = True
            vec1 = self.getvec(o, main_atom)  #[o.x-main_atom.x, o.y-main_atom.y, o.z-main_atom.z]
            for w in check_water:
                vec2 = self.getvec(w, main_atom)  #[w.x-main_atom.x, w.y-main_atom.y, w.z-main_atom.z]
                if (self.angle(vec1, vec2) < 10):
                    safe = False
            if (safe):
                ret.append(o)
        return ret


    # merges close atoms in mod into one average point.
    def merge_close(self, mod, dist):
        newmod = []
        for a in mod:
            close = False
            for i, b in enumerate(newmod):
                if (self.distance(a, b) < dist):
                    newmod[i] = self.get_avg_point([a, b])
                    close = True
            if not close:
                newmod.append(a)
        return newmod

    # places optimal atoms around all charged groups. considers metal
    # coordination bonds and possible important water molecules.
    def place_optimals(self, lining_atoms, metals, pocket, radius, protein_atoms, check_water, wat_conversion, chargerad, acceptor, H_add, charge_lib, angle_tolerance=20):
        metal_coordinate_atoms = set()
        for m in metals:
            for atom in m.bonds:
                metal_coordinate_atoms.add(atom)
        self.print_this("ver-1", "Placing charged spots");


        temp = self.get_add_lines(metal_coordinate_atoms, lining_atoms, acceptor, H_add)
        addery = temp[0]
        del_set = temp[1]
        done_set = temp[2]

        temp = self.do_lines(pocket, addery, protein_atoms, del_set, acceptor, H_add, radius, check_water, angle_tolerance, done_set, chargerad, charge_lib)
        circles = temp[0]
        optionals = temp[1]
        no_recheck = temp[2]

        # TODO: This is never used. Remove or figure out the original purpose.
        if wat_conversion:
            tempcir = []+circles
            #circles = self.new_waters(circles, wat_conversion, acceptor, H_add)
            for key in optionals:
                tempcir.extend(optionals[key])
            # There is an easier way to accomplish the following, but for now
            # this'll do.
            tempcir = self.new_waters(tempcir, wat_conversion, acceptor, H_add)
            deldat = tempcir[1]
            tempcir = tempcir[0]

            new_opt = {}

            for entr in deldat:
                new_circ = []
                for c in tempcir[entr[0]]:
                    if not c.idn == entr[1].idn:
                        new_circ.append(c)
                tempcir[entr[0]] = new_circ
            new_temp = []

            for index, line in enumerate(tempcir):
                count = 1
                for indx, lin in enumerate(tempcir, start=index):
                    if (len(lin) > 0 and len(line) > 0):
                        if(lin[0].resseq == line[0].resseq):
                            count += 1
                            if (lin[0].resseq not in new_opt):
                                new_opt[str(lin[0].resseq)] = []
                            new_opt[str(lin[0].resseq)].append(lin)
                if (count < 2):
                    new_temp.append(line)
                else:
                    new_opt[str(line[0].resseq)].append(line)
            tempcir = new_temp

            for key in new_opt:
                largest = [60000, None]
                others = []
                for mod in new_opt[key]:
                    if (len(mod) < largest[0]):
                        if (largest[1] is not None):
                            others.append(largest[1])
                        largest = [len(mod), mod]
                    else:
                        others.append(mod)
                if (len(others) > 0):
                    new_opt[key] = others
                tempcir.append(largest[1])
                optionals[key] = new_opt[key]
            circles = tempcir

        if len(del_set) > 0:
            for idx, mod in enumerate(circles):
                newmod = []
                addset = set()
                for a in mod:
                    add = True
                    for b in del_set:
                        if self.distance(a, b[0]) <= b[1]:
                            add = False
                            addset.add(b[0])
                    if add:
                        newmod.append(a)
                newmod.extend(list(addset))
                circles[idx] = newmod
            for key, value in optionals.items():
                newval = []
                for mod in value:
                    newmod = []
                    for a in mod:
                        for b in del_set:
                            if self.distance(a, b[0]) > b[1]:
                                newmod.append(a)
                    newval.append(newmod)
                optionals[key] = newval

        # TODO: This is never used. Remove or figure out the original purpose.
        if check_water:
            for w in check_water:
                if (w.elemen.lower() == "h"):
                    continue
                if ((len(w.acceptor_bonds) + len(w.donor_bonds)) >= 4 or (len(w.acceptor_bonds) + len(w.donor_bonds)) < 2):
                    continue

                refset = w.acceptor_bonds+w.donor_bonds
                donors = w.donor_bonds
                donors_occupied = len(donors)
                acceptors_occupied = 0

                for a in refset:
                    if a.elemen.lower() == "n":
                        acceptors_occupied += 1

                circle = self.get_circle(w, refset[0], 109.5, acceptor, 5, radius)
                first = self.get_closest(refset[1], circle)
                f_idx = first[2]

                second_atom = circle[int(f_idx - (len(circle)/3))]
                third_atom = circle[int(f_idx - ((len(circle)/3)*2))]
                # if both atoms will be donors, move them further a tad.
                second_farther = self.get_to_from(second_atom, w, acceptor + H_add, radius)
                third_farther = self.get_to_from(third_atom, w, acceptor + H_add, radius)
                onemod = False
                if (donors_occupied == 2):
                    onemod = True
                    atm = [second_farther, third_farther]
                elif (acceptors_occupied == 2):
                    onemod = True
                    atm = [second_atom, third_atom]
                else:
                    mods = [
                            [second_atom, third_farther],
                            [third_atom, second_farther]
                            ]
                if (onemod):
                    for a in atm:
                        a.radius = chargerad
                    circles.append(atm)
                else:
                    for m in mods:
                        for a in m:
                            a.radius = chargerad
                    optionals[str(w.resseq)] = mods
        return [circles, optionals, no_recheck]


    # Fixes overlap in mod. overlap = two points within limit of one another.
    def fix_overlap(self, mod, limit=0.5):
        newmod = []
        avg = self.get_avg_point(mod)
        for i, a in enumerate(mod):
            a_avg = self.distance(a, avg)
            overlap = False
            if len(newmod) > 0:
                for j, b in enumerate(newmod):
                    b_avg = self.distance(b, avg)
                    if self.distance(a, b) < limit:
                        overlap = True
                        ch = []
                        if a.tempf != 0.0:
                            ch.append(a)
                        if b.tempf != 0.0:
                            ch.append(b)
                        if len(ch) == 0:
                            if a_avg < b_avg:
                                newmod.append(a)
                            else:
                                newmod.append(b)
                        elif len(ch) == 1:
                            newmod[j] = (ch[0])
                        else:
                            if self.sign(a.tempf) == self.sign(b.tempf):
                                if math.fabs(a.tempf) > math.fabs(b.tempf):
                                    newmod[j] = a
                                else:
                                    newmod[j] = b
                            else:
                                newmod[j] = a
                if not overlap:
                    newmod.append(a)
            else:
                newmod.append(a)
        return self.deldupl(newmod)

    # deletes duplicates from seq.
    def deldupl(self, seq):
        seen = set()
        seen_add = seen.add
        return [ x for x in seq if x not in seen and not seen_add(x)]

    # true if iterable is not empty, or a list(etc) of empty lists(etc)
    def not_empty(self, iterable):
        if len(iterable) > 0:
            for r in iterable:
                if hasattr(r, '__iter__'):
                    return self.not_empty(r)
                else:
                    return True
        return False

    # returns atoms from pocket that are close enough to center_atom, or to
    # other atoms that are close enough to center_atom (recursive function)
    def remove_gaps(self, center_atom, pocket, distance_limit):
        safe = set([center_atom])
        lastadd = True
        while(lastadd):
            added = 0
            for a in self.get_within_radius_of_list(distance_limit, pocket, list(safe)):
                if a not in safe:
                    safe.add(a)
                    added += 1
            if added > 0:
                lastadd = True
            else:
                lastadd = False
        return list(safe)

    # true if atom is in the list of polar atoms
    def is_polar(self, atom, pol_list=["o", "n", "p", "cl", "f"]):
        polset = set(pol_list)
        if atom.elemen.lower() in polset:
            return True
        return False

    # returns planar set of atoms around atom from circle
    def get_planar(self, atom, circle, plane):
        tempdata = []
        ret = []
        for c_a in circle:
            arr = [self.getvec(c_a, atom)]
            for a in plane:
                arr.append(self.getvec(a, atom))
            det = self.array_det(arr)
            tempdata.append([c_a, math.fabs(det)])
        tempdata.sort(key=lambda x: x[1])
        ret.append(tempdata[0][0])
        ret.append(self.get_farthest_from(ret[0], circle)[0])
        ret.extend(plane)
        return ret

    # returns three atoms in tetrahedral arrangement around atom, based
    # reference atom. circle = a circle of atoms in which the three remaining
    # points can be found.
    def get_tetra(self, atom, circle, ref):
        increment = int(len(circle)/3)
        first = self.get_closest(ref, circle)
        second = circle[first[2]-increment]
        third = circle[first[2]-(increment*2)]
        first = first[0]
        return [first, second, third]

    def find_h_bond_candidates(self, atom, nearby):
        bonds = []
        for protatom in nearby:
            if not protatom.resseq == atom.resseq:
                if protatom.elemen.lower() == "n":
                    if len(protatom.bonds) >= 3:
                        bonds.append([protatom, "donor"])
                    else:
                        bonds.append([protatom, "acceptor"])
                elif protatom.elemen.lower() == "o":
                    if len(protatom.bonds) < 2:
                        bonds.append([protatom, "acceptor"])
                    else:
                        bonds.appenc([protatom, "no clue"])
        atom.new_H_bonds = bonds



    def get_oh(self, atom, ref, angle, h_bond_ref, acc_distance, H_add, nearby, force_multi=False):
        h_mod = False
        h_ref = None
        for a in atom.bonds:
            if a.elemen.lower() == "h":
                h_mod = True
                h_ref = a
        if (h_mod):
            cir = self.get_circle(atom, ref, angle, acc_distance, 5, 0.5)
            h_spot = self.get_closest(h_ref, cir)
            increment = int(len(cir)/3)
            mod = [self.get_to_from(h_spot[0], atom, self.distance(atom, h_ref)+H_add, 0.5), cir[h_spot[2]-increment], cir[h_spot[2]-(2*increment)]]
            return [False, mod]

        else:
            ref2=h_bond_ref
            don = []
            acc = []
            no_clue = []
            self.find_h_bond_candidates(a, nearby)

            for a in atom.new_H_bonds:
                a_type = a[1]
                if a_type == "acceptor":
                    acc.append(a[0])
                elif a_type == "donor":
                    don.append(a[0])
                else:
                    no_clue.append(a[0])

            multimod = False
            acc_mod = False
            don_mod = False
            distance = acc_distance
            if len(don) >= 1:
                acc_mod = True
                don_spot_refs = [don[0]]
            elif len(acc) == 2:
                distance = acc_distance + H_add
                don_spot_refs = acc
                don_mod = True
            else:
                multimod = True
            if force_multi:
                multimod=True

            cir = self.get_circle(atom, ref, angle, distance, 5, 0.5) # is this useless?
            first = self.get_closest(ref2, cir)
            indx = first[2]
            increment = int(len(cir)/3)
            first = first[0]

            if multimod:
                if distance == acc_distance:
                    close_cir = cir
                    far_cir = None
                    distance += H_add
                else:
                    close_cir = None
                    far_cir = cir
                    distance -= H_add
                cir2 = self.get_circle(atom, ref, angle, distance, 5, 0.5)
                if far_cir is None:
                    far_cir = cir2
                else:
                    close_cir = cir2

                close_ref = self.get_closest(first, close_cir)
                close_inc = int(len(close_cir)/3)
                close_three = [close_ref[0], close_cir[close_ref[2]-close_inc], close_cir[close_ref[2]-(2*close_inc)]]

                far_ref = self.get_closest(first, far_cir)
                far_inc = int(len(far_cir)/3)
                far_three = [far_ref[0], far_cir[far_ref[2]-far_inc], far_cir[far_ref[2]-(2*far_inc)]]

                opt = [
                       [close_three[0], close_three[1], far_three[2]],
                       [close_three[0], far_three[1], close_three[2]],
                       [far_three[0], close_three[1], close_three[2]]
                       ]

                return [True, opt]
            else:
                mod = []
                if acc_mod:
                    donspot = self.get_closest(don_spot_refs[0], cir)
                    acc_spots = [cir[donspot[2]-increment], cir[donspot[2]-(2*increment)]]
                    mod = acc_spots
                elif don_mod:
                    donspot = self.get_farthest_from(self.get_avg_point(don_spot_refs), cir)[0]
                    mod = [donspot]
                return [False, mod]

    # returns True if candidate (an atomlist) is not included in compare_list,
    # which is a list of atom lists. if subset_is_same =True, also returns true
    # is candidate is a subset of some atom list in compare_list.
    def not_included(self, candidate, compare_list, subset_is_same=True):

        if (len(compare_list) < 1):
            return [True, candidate]
        comparison_list = []
        for atom in candidate:
            if abs(atom.tempf) > 0.1:
                comparison_list.append(atom)
        for idx, modentry in enumerate(compare_list):
            similarity_count = 0
            for atom in comparison_list:
                for matom in modentry:
                    if self.distance(matom, atom) < 0.5:
                        if abs(matom.tempf) > 0.1:
                            if abs(abs(matom.tempf) - abs(atom.tempf)) < 0.2:
                                similarity_count += 1
            if similarity_count == len(comparison_list):
                if len(candidate) == len(modentry):
                        return [False, 0, -1]
                if subset_is_same:
                    if len(candidate) <= len(modentry):
                        return [False, list(modentry), -1]
                    elif len(modentry) <= len(candidate):
                        return [False, list(candidate), idx]
        return [True, list(candidate)]

    # returns true if candidate is not in compare_list. candidate = a model.
    # that is, a set of atoms. comparison based on coordinates. if even one
    # point does not match = True
    def included(self, candidate, compare_list, dist_tol=0.3):
        if (len(compare_list) < 1):
            return [False, None, 0]

        for idx, mod in enumerate(compare_list):
            tally_ho = 0.0
            for atom in candidate:
                temp = self.get_closest(atom, mod)
                tally_ho += temp[1]
            if tally_ho <= dist_tol:
                return [True, mod, idx]
        return [False, None, 0]

    # sets charge in pocket atoms to charges based on the charges of protein
    # atoms that are distance away from them at the most.
    def set_ch(self, pocket, protein, distance):
        for a in pocket:
            chsum = 0.0
            sumnum = 0
            for p in protein:
                if self.distance(p, a) < distance:
                    chsum = chsum + p.new_ch
                    sumnum += 1
            if (sumnum > 0):
                a.tempf = -(chsum/sumnum)

    # returns atoms from protein and hetatoms that have coordinate bonds with
    # atoms. atom_info[0] = distance for coordinate bonds.
    def get_coord_bonds(self, atom, protein, hetatoms, atom_info):
        distance = float(atom_info[0])
        limit_dist = float(atom_info[1].strip("()"))
        conf_num = 6

        if atom_info[2] == "octa":
            conf_num = 6
        elif atom_info[2] == "tribipy":
            conf_num = 5

        bonds = []
        found_num = 0
        testset = []+protein+hetatoms
        distdata = []
        for patom in testset:
            if (patom.idn != atom.idn):
                if (patom.elemen.lower() != "h"):
                    distdata.append([patom, self.distance(atom, patom)])
        distdata.sort(key=lambda x: x[1])
        distdata2=[]
        #s.print_this("war", conf_num)
        for d in distdata:
            if found_num >= (conf_num):
                break
            if (d[1] <= distance):
                bonds.append(d[0])
                distdata2.append(d)
                found_num +=1
            elif found_num < (conf_num - 1) and d[1] < limit_dist:
                distdata2.append(d)
                found_num +=1
                bonds.append(d[0])
        return bonds

    # returns atoms from "possibles" within "distance" of any atom(s) from
    # "core". possibles = a list of lists of atoms.
    def get_within_radius_of_set(self, distance, possibles, core):
        ret = set()
        for atom in core:
            for setti in possibles:
                for atom2 in setti:
                    if (self.distance(atom, atom2) < distance):
                        ret.add(atom2)
                        continue
        return list(ret)

    # returns atoms from "possibles" within "distance" of any atom(s) from
    # "core". possibles = list
    def get_within_radius_of_list(self, distance, possibles, core):
        ret = set()
        for atom in core:
            for atom2 in possibles:
                if (self.distance(atom, atom2) < distance):
                    ret.add(atom2)
                    continue
        return list(ret)


    def get_points(self, atom, plane_ref, plane_ref2, distance, radius):
        points = []
        cir = self.get_circle(atom, plane_ref, 90, distance, 5, radius)
        candidates = []
        for c_a in cir:
            arr = [self.getvec(c_a, atom), self.getvec(plane_ref, atom), self.getvec(plane_ref2, atom)]
            det = self.array_det(arr)
            candidates.append([c_a, math.fabs(det)])
        candidates.sort(key=lambda x: x[1], reverse=True)
        for a in candidates[:2]:
            points.append(a[0])
        return points

    # returns a plane of three atoms at distance from main_atom.
    # points = point[0] should be an atom in the direction where the plane
    # should be. planepoints = any points already on the plane.
    def three_plane(self, points, distance, planepoints, main_atom, radius):
        cir = self.get_circle(main_atom, points[0], 90, distance, 5, radius)
        increment = int(len(cir)/3)
        if len(planepoints) == 1:
            planepoint = planepoints[0]
            first_cir = self.get_closest(planepoint, cir)
            second_cir = cir[first_cir[2] - increment]
            third_cir = cir[first_cir[2] - (increment*2)]
            first_cir = first_cir[0]
        else:
            first_cir = self.get_closest(planepoints[0], cir)[0]
            second_cir = self.get_closest(planepoints[1], cir)[0]
            third_cir = self.get_farthest_from(self.get_avg_point([first_cir, second_cir]), cir)[0]

        return [first_cir, second_cir, third_cir]

    # returns vector cross product
    def vector_cross_product(self, a, b):
        c = [a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0]]
        return c

    # creates optimal atoms around metal atoms.
    def place_optimal_metals(self, metal_atoms, geo_lib, radius, to_check, chargerad, pocket, distance=None):
        final_atoms = []
        orig_distance = distance
        for atom in metal_atoms:
            my_optimals = []
            if orig_distance is None and atom.elemen.lower() not in geo_lib :
                self.print_this("err", "No bond distance defined for {} of {}{}\n Ignoring atom.".format(atom.name, atom.resname, atom.resseq))
                continue
            if len(atom.bonds) < 1:
                self.print_this("err", "Not enough coordination atoms found for {} of {}{}\n Ignoring atom.".format(atom.name, atom.resname, atom.resseq))
                continue
            configuration = geo_lib[atom.elemen.lower()][2]
            # New: If distance was not given and it is found in library, use
            # library distance. Otherwise use parameter distance
            if orig_distance is None and len(geo_lib[atom.elemen.lower()] ) > 2:
                #if len(geo_lib[atom.elemen.lower()] ) > 2:
                distance = float(geo_lib[atom.elemen.lower()][0])
            else:
                distance = orig_distance
            # Entry is added to atom.addline, and used only in function
            # check_again. What is its purpose?
            entry = [atom, "", 180, "acceptor", "linear"]
            if (configuration == "octa"):
                if (len(atom.bonds) == 5):
                    for a in atom.bonds:
                        if a.elemen.lower() != "n":
                            entry[1] = copy(a) # Copy or alias (as it was originally)?
                if atom.resname.lower() == "hem":
                    refs = []
                    sulph = None
                    # Find heme important bonds
                    for b_a in atom.bonds:
                        if (b_a.resname.lower() == "hem" and
                        (b_a.name.lower() == "na"
                        or b_a.name.lower() == "nb"
                        or b_a.name.lower() == "nc"
                        or b_a.name.lower() == "nd")):
                            refs.append(copy(b_a))
                        elif (b_a.name.lower() == "sg" # Cys
                        or b_a.name.lower() == "ne2"): # His
                            sulph = copy(b_a)
                    # Build atoms on normal vectors for all planes formed by all
                    # nitrogen pairs and heme metal.
                    if len(refs) > 1 and sulph is not None:
                        # a list containing all combination tuples of the
                        # reference atoms.
                        vector_combinations = list(combinations(refs,2))
                        heme_normals = []
                        for refTuple in vector_combinations:
                            vec1 = self.getvec(refTuple[0], atom)
                            vec2 = self.getvec(refTuple[1], atom)
                            vector_normal_to_both = self.vector_cross_product(vec1, vec2)
                            norm_atom = copy(atom)
                            norm_atom.x = norm_atom.x+vector_normal_to_both[0]
                            norm_atom.y = norm_atom.y+vector_normal_to_both[1]
                            norm_atom.z = norm_atom.z+vector_normal_to_both[2]
                            # If the atom is towards coordinated amino acid and
                            # not substrate binding site, move atom towards
                            # binding site.
                            if self.distance(sulph, norm_atom) < self.distance(norm_atom, atom):
                                norm_atom = self.get_to_from(atom, norm_atom, 4 + self.distance(atom, norm_atom), radius)
                            heme_normals.append(norm_atom)
                        # Place optimal coordination point along the "nitrogen
                        # pair plane normal" average
                        my_optimals = [self.get_to_from(self.get_avg_point(heme_normals), atom, distance, radius)]
                    else:
                        self.print_this("war", "Metal {} in {} {}: Ignoring. No amino acid coordination atom or enough heme nitrogens found! Check their atom names.".format(atom.name, atom.resname, atom.resseq))
                # Other metal atoms
                elif (len(atom.bonds) > 0):
                    ref1 = atom.bonds[0]
                    ref1dist = self.distance(ref1, atom)
                    ref1vec = self.getvec(atom, ref1)
                    ref2 = None
                    planerefdat = None
                    plane = []
                    planeref = []
                    for a in atom.bonds:
                        if a is not ref1:
                            avec = self.getvec(atom, a)
                            if 195 > self.angle(avec, ref1vec) > 165: #180 degree angle
                                ref2 = a
                            else:
                                planeref.append(a)
                    if ref2 is None:
                        ref2 = self.get_to_from(atom, ref1, 2*ref1dist, radius)
                    points = [copy(ref1), copy(ref2)]
                    cir = self.get_circle(atom, ref1, 90, ref1dist, 5, radius)
                    if len(planeref) > 0:
                        planerefdat = self.get_closest(planeref[0], cir)
                    else:
                        planerefdat = self.get_closest(self.get_avg_point(to_check), cir)
                    increment = int(len(cir)/4)
                    plane = [planerefdat[0], cir[(planerefdat[2]-increment)], cir[(planerefdat[2]-2*increment)], cir[(planerefdat[2]-3*increment)]]
                    my_optimals = points+plane
                    entry[1] = self.get_farthest_from(self.get_avg_point(pocket), points)[0]
                else:
                    self.print_this("err", "Not enough coordination atoms found for {} of {}{}\n Ignoring atom.".format(atom.name, atom.resname, atom.resseq))
            elif (configuration == "tribipy"):
                contin = True
                if len(atom.bonds) < 1:
                    self.print_this("err", "Not enough coordination atoms found for {} of {}{}\n Ignoring atom.".format(atom.name, atom.resname, atom.resseq))
                    contin = False
                elif len(atom.bonds) < 2:
                    self.print_this("war", "Not enough coordination atoms found for {} of {}{}\nMaking a guess...".format(atom.name, atom.resname, atom.resseq))
                    atom.bonds.append(self.get_to_from(atom, atom.bonds[0], 2*self.distance(atom, atom.bonds[0]), radius))
                if contin:
                    ref1 = atom.bonds[0]
                    ref2 = atom.bonds[1]
                    ref1dist = self.distance(atom, ref1)
                    #ref2dist = self.distance(atom, ref2)
                    ref1vec = self.getvec(ref1, atom)
                    ref2vec = self.getvec(ref2, atom)

                    ref_ang = self.angle(ref1vec, ref2vec)

                    plane = None
                    points = None
                    ref3 = None
                    ref3vec = None
                    if (len(atom.bonds) > 2):
                        ref3 = atom.bonds[2]
                        ref3vec = self.getvec(ref3, atom)

                    if (135 >= ref_ang >= 105): #120 degree angle
                        if ref3 is None:
                            ref3 = self.get_points(atom, ref1, ref2, ref1dist, radius)[0]
                            ref3vec = self.getvec(atom, ref3)
                        if self.angle(ref1vec, ref3vec) <= 105: # 90 degree angle
                            points = [ref3, self.get_to_from(atom, ref3, 2*self.distance(atom, ref3), radius)]
                            plane = self.three_plane(points, ref1dist, [ref1, ref2], atom, radius)
                        else:
                            plane = [ref1, ref2, ref3]
                            points = self.get_points(atom, ref1, ref2, ref1dist, radius)
                    elif (135 < ref_ang): #180 degree angle
                        points = [ref1, ref2]
                        if ref3 is None:
                            ref3 = self.get_closest(atom, to_check)
                        plane = self.three_plane(points, ref1dist, [ref3], atom, radius)
                    elif (105 > ref_ang): # 90 degree angle
                        if ref3 is None:
                            ref3 = self.get_to_from(atom, ref1, 2*self.distance(atom, ref1), radius)
                            ref3vec = self.getvec(atom, ref3)
                        if self.angle(ref1vec, ref3vec) > 165: #180 degree angle
                            points = [ref1, ref3]
                            plane = self.three_plane(points, ref1dist, [ref2], atom, radius)
                        elif self.angle(ref1vec, ref3vec) >= 105: #120 degree angle
                            points = [ref2, self.get_to_from(atom, ref2, 2*self.distance(atom, ref2), radius)]
                            plane = self.three_plane(points, ref1dist, [ref1, ref3], atom, radius)
                        elif self.angle(ref1vec, ref3vec) < 105:
                            points = [ref2, ref3]
                            plane = self.three_plane(points, ref1dist, [ref1], atom, radius)
                    entry[1] = self.get_farthest_from(self.get_avg_point(pocket, points))[0]
                    my_optimals = plane + points
            for a in my_optimals:
                a.resname = "MET"
                a.resseq = atom.resseq
                a.elemen = "O"
                a.name = atom.name
                a.host = atom
                a.tempf = -atom.new_ch
                atom.kiddies.append(a)
                a.addline = entry
                a.radius = chargerad
            final_atoms.append(my_optimals)
        return final_atoms

    # True for positive (INCLUDING 0.0), False for negative
    def sign(self, number):
        if (type(number) is not float):
            number = float(number)
        return number >= 0.0

    # deletes atoms from "pocket" that are on the same side as "center" of a
    # plane defined by "planepoints"
    def plane_exclude(self, center, planepoints, pocket):
        cent = [center.x - planepoints[0].x, center.y - planepoints[0].y, center.z - planepoints[0].z]
        p1 = self.getvec(planepoints[1], planepoints[0])   #[planepoints[1].x - planepoints[0].x, planepoints[1].y - planepoints[0].y, planepoints[1].z - planepoints[0].z]
        p2 = self.getvec(planepoints[2], planepoints[0])   # [planepoints[2].x - planepoints[0].x, planepoints[2].y - planepoints[0].y, planepoints[2].z - planepoints[0].z]
        cen_det = self.array_det([cent, p1, p2])
        center_sign = self.sign(cen_det)

        approved = []
        for p_atom in pocket:
            p_arr = self.getvec(p_atom, planepoints[0])   #[p_atom.x - planepoints[0].x, p_atom.y - planepoints[0].y, p_atom.z - planepoints[0].z]
            p_det = self.array_det([p_arr, p1, p2])
            candidate_sign = self.sign(p_det)
            if (candidate_sign is center_sign):
                approved.append(p_atom)
        return approved

    # vector from fromatom to toatom in the form of a list [x, y, z]
    def getvec(self, toatom, fromatom):
        return [toatom.x - fromatom.x, toatom.y - fromatom.y, toatom.z - fromatom.z]


    # counts how many angles between atom and atoms in bonded are in
    # angle+/-tolerance range
    def count_angles(self, atom, bonded, angle, tolerance):
        count = 0
        if len(bonded) >= 2:
            ref = bonded[0]
            refvec = self.getvec(ref, atom)
            for a in bonded[1:]:
                v = self.getvec(a, atom)
                if angle + tolerance > self.angle(v, refvec) > angle - tolerance:
                    count += 1
        return count

    # adds optimal spots for a given residue
    def optimal_for_res(self, residue, ch_atom_radius, ang_lib, acc_distance, h_add, all_atoms):
        fin_opt = []
        fin_options = {}
        resname = residue[0].resname.lower()
        if resname in ang_lib:
            for atom in residue:
                opt_resnum = str(self.start_own_resnums)
                entry = [atom, atom, 180, "acceptor", "tetrahedral"]
                excl_range = self.same_residue_distance_tolerance
                if atom.name.lower() in ang_lib[resname]:
                    opt = []
                    mod = []
                    data = ang_lib[resname][atom.name.lower()]
                    angle = float(data[0])
                    entry[2] = angle
                    conf = data[1]
                    dist = data[2]
                    condition = None
                    if len(data) > 3:
                        condition = data[3]
                    true_distance = acc_distance
                    if condition is not None:  # data = [{if_h_then_109.5, tetra, oh}]
                        #~ print("::::::::::::")
                        #~ print(condition)
                        #~ print(data)
                        condata = condition.strip("{}").split("_then_")
                        #~ print(condata)
                        con_element = condata[0].split("_")[1]
                        condata = condata[1].split(",")
                        con = False
                        for a in atom.bonds:
                            if a.elemen.lower() == con_element:
                                con = True
                                break
                        if con:
                            #~ print(condata)
                            angle = float(condata[0])
                            #~ conf = condata[1]
                            #~ dist = condata[2]

                    if dist == "don":
                        true_distance += h_add
                        entry[3] = "donor"

                    ref1 = atom.bonds[0]
                    for a in atom.bonds:
                        if a.elemen.lower() != "h":
                            ref1 = a
                            break
                    cir = self.get_circle(atom, ref1, angle, true_distance, 5, ch_atom_radius)

                    if conf == "planar":
                        entry[4] = "planar"
                        plane_refs = set([])
                        for a in atom.bonds:
                            if a.elemen.lower() != "h" and a not in plane_refs:
                                plane_refs.add(a)
                        plane_refs = list(plane_refs)
                        if len(plane_refs) < 1:
                            break
                        elif len(plane_refs) < 2:
                            for a in plane_refs[0].bonds:
                                if a.elemen.lower() != "h" and a != atom and a not in plane_refs:
                                    plane_refs.append(a)
                                    break
                        if len(plane_refs) > 2:
                            plane_refs = self.get_closest(atom, plane_refs)
                            new_planref = [plane_refs[0], plane_refs[3]]
                            plane_refs = new_planref
                        entry[1] = self.get_avg_point(plane_refs)
                        mod = self.get_planar(atom, cir, plane_refs)

                    elif dist == "oh":
                        entry[4] = "oh"
                        h_ref = self.get_closest(atom, residue)[0]
                        h_ref = self.get_to_from(atom, h_ref, (2*self.distance(atom, h_ref)), 0.5)
                        entry[1] = h_ref
                        opt = self.get_oh(atom, ref1, angle, h_ref, acc_distance, h_add, self.get_within_radius(atom, all_atoms,acc_distance + h_add))
                        if not opt[0]:
                            mod = opt[1]
                            opt = []
                        else:
                            opt = opt[1]

                    elif conf == "tetra":
                        entry[4] = "tetrahedral"
                        entry[1] = ref1
                        for bonda in atom.bonds:
                            if bonda != ref1 and bonda != atom:
                                ref1 = bonda
                                break
                        mod = self.get_tetra(atom, cir, ref1)

                    if (len(mod) > 0):
                        newmod = []
                        for o_app in mod:
                            add = True
                            angvec = self.getvec(o_app, atom)
                            for a in residue:
                                if a != atom:
                                    if self.distance(a, o_app) < excl_range:
                                        add = False
                                        break
                                    secvec = self.getvec(a, atom)
                                    if self.angle(angvec, secvec) < 10 and self.distance(a, o_app) < acc_distance:
                                        add = False
                                        break
                            if add:
                                o_app.resname = atom.resname
                                o_app.resseq = atom.resseq
                                o_app.radius = ch_atom_radius
                                o_app.addline = entry
                                if self.distance(o_app, atom) > acc_distance:
                                    o_app.tempf = atom.new_ch
                                else:
                                    o_app.tempf = -atom.new_ch
                                newmod.append(o_app)
                        if len(newmod) > 0:
                            fin_opt.extend(newmod)
                    if len(opt) > 0:
                        new_opt = []
                        for i, mon in enumerate(opt):
                            new_entry = []
                            for a in mon:
                                angvec = self.getvec(a, atom)
                                add = True
                                for a2 in residue:
                                    if a2 != atom:
                                        if self.distance(a2, a) < excl_range:
                                            add = False
                                            break
                                        secvec = self.getvec(a2, atom)
                                        if self.angle(angvec, secvec) < 10:
                                            add = False
                                            break
                                if add:
                                    a.resname = "HER"
                                    a.resseq = a.resseq
                                    a.radius = ch_atom_radius
                                    a.addline = entry
                                    if self.distance(a, atom) > acc_distance:
                                        a.tempf = atom.new_ch
                                    else:
                                        a.tempf = -atom.new_ch
                                    new_entry.append(a)
                            if len(new_entry) > 0:
                                new_opt.append(new_entry)
                        if len(new_opt) > 0:
                            fin_options.update({opt_resnum: opt})
                            self.start_own_resnums += 1
        else:
            chd = []
            for a in residue:
                if self.is_charged(a):
                    chd.append(a)
            for a in chd:
                entry = [a, a, 180, "acceptor", "tetrahedral"]
                opt = []
                o_app = None
                #ch = 0.0
                ang = 180
                ref = None
                # TODO: Test that the atom really has bonds before trying to access.
                #       Probably skip atoms that do not have bonds, and create their
                #       coordination points elsewhere (water, metal atoms).
                if a.elemen.lower() == "h":
                    ref = a.bonds[0]
                    o_app = self.get_to_from(a, ref, self.distance(a, a.bonds[0]) + 1, 0.7)
                    entry[1] = ref
                    entry[4] = "linear"
                    o_app.tempf = -a.new_ch
                    opt.append(o_app)
                else:
                    if a.elemen.lower() == "n" or a.elemen.lower() == "p":
                        entry[3] = "donor"
                        dist = acc_distance + h_add
                    else:
                        dist = acc_distance
                    non_h = []
                    h = []
                    for ba in a.bonds:
                        if ba.elemen.lower() == "h":
                            h.append(ba)
                        else:
                            non_h.append(ba)
                    if (a.elemen.lower() == "n"):
                        ang = 106.7
                        if (len(non_h) == 2):
                            ang = 120
                    elif (a.elemen.lower() == "p"):
                        ang = 100.0
                    elif self.count_angles(a, a.bonds, 109.5, 5) > 0:
                        ang = 109.5
                    else:
                        ang = 120
                        entry[4] = "planar"
                    entry[1] = a.bonds[0]
                    cir = self.get_circle(a, a.bonds[0], ang, dist, 5, 0.7)
                    if ang == 109.5 and a.elemen.lower() == "o":
                        cir2 = self.get_circle(a, a.bonds[0], ang, dist+1.0, 5, 0.7)

                        ref = None
                        if len(h) > 0:
                            ref = self.get_closest(h[0], cir2)
                        elif len(non_h) > 2:
                            for atm in non_h:
                                if atm.idn != a.bonds[0].idn:
                                    ref = self.get_closest(atm, cir)
                                    break
                        if ref is None:
                            ref = self.get_closest(self.get_farthest_from_avg(cir2, residue)[0], cir2)

                        incr = int(len(cir2)/3)
                        one = ref[0]
                        one.tempf = a.new_ch
                        two = self.get_closest(cir2[ref[2]-incr], cir)[0]
                        two.tempf = -a.new_ch
                        three = self.get_closest(cir2[ref[2]-(incr*2)], cir)[0]
                        three.tempf = -a.new_ch
                        opt.extend([one, two, three])
                    elif ang == 106.7 or ang == 100.0:
                        ref = None
                        if len(h) > 0:
                            ref = self.get_closest(h[0], cir2)
                        elif len(non_h) > 2:
                            for atm in non_h:
                                if atm.idn != a.bonds[0].idn:
                                    ref = self.get_closest(atm, cir)
                                    break
                        if ref is None:
                            ref = self.get_closest(self.get_farthest_from_avg(cir, residue)[0], cir)

                        incr = int(len(cir)/3)
                        one = ref[0]
                        one.tempf = a.new_ch
                        two = cir[ref[2]-incr]
                        two.tempf = a.new_ch
                        three = cir[ref[2]-(incr*2)]
                        three.tempf = a.new_ch
                        opt.extend([one, two, three])

                    elif ang == 120:
                        ref = [a.bonds[0], 0]
                        ref2 = a.bonds[0].bonds[0]
                        candidates = []
                        #dun = []
                        for c_a in cir:
                            arr = [self.getvec(c_a, a), self.getvec(ref[0], a), self.getvec(ref2, a)]
                            det = self.array_det(arr)
                            candidates.append([c_a, math.fabs(det)])
                        candidates.sort(key=lambda x: x[1])
                        for a2 in candidates[:2]:
                            opt.append(a2[0])
                            a2[0].tempf = -a.new_ch

                newmod = []
                for o_app in opt:
                    add = True
                    angvec = self.getvec(o_app, a)
                    for at in residue:
                        if at != a:
                            if self.distance(at, o_app) < self.same_residue_distance_tolerance:
                                add = False
                                break
                            secvec = self.getvec(at, a)
                            if self.angle(angvec, secvec) < 10:
                                add = False
                                break
                    if add:
                        o_app.resname = a.resname
                        o_app.resseq = a.resseq
                        o_app.radius = ch_atom_radius
                        o_app.addline = entry
                        if self.distance(o_app, a) > acc_distance:
                            o_app.tempf = a.new_ch
                        else:
                            o_app.tempf = -a.new_ch
                        newmod.append(o_app)
                if len(newmod) > 0:
                    fin_opt.extend(opt)
        return [fin_opt, fin_options]

    # rechecks disqualified circle atoms and tries to find non-disqualified
    # replacements by varying hydrogen bond angles or distances to less optimal
    # values.
    def check_again(self, cir_atoms, lining, pocket, flex_angle, radius, no_recheck, chargerad, h_dist, h_add):
        ret = []
        distance = h_add
        #H_add = 1
        max_len_add = 0.8
        for atom in cir_atoms:
            env_lin = self.get_within_radius(atom, lining, 5)
            env_poc = self.get_within_radius(atom, pocket, 5)
            if (len(atom.addline) < 2):
                continue
            if not ((len(env_lin) < 1) or len(env_poc) < 1):
                add_line = atom.addline
                main_atom = add_line[0]
                if main_atom in no_recheck:
                    continue
                ref_atom = add_line[1]
                angle = add_line[2]
                if (angle == 180 or angle == 0):
                    continue
                distance = self.distance(atom, main_atom)

                gotcha = False
                messiah = None
                #done = False
                i = 0.0
                while i < max_len_add:
                    candidate = self.get_to_from(atom, main_atom, distance + i, radius)
                    if (not self.no_overlap(candidate, pocket, cheat = 1.0)[0]):
                        if (self.no_overlap(candidate, lining)[0]):
                            gotcha = True
                            messiah = candidate
                            break
                    i = i + 0.1
                if (gotcha):
                    ret.append(messiah)
                    messiah.tempf = atom.tempf
                    messiah.host = main_atom
                else:
                    circledump = []
                    for i in range(int(angle), int(angle + flex_angle), 5):
                        circledump.extend(self.get_circle(main_atom, ref_atom, i, distance, 5, radius))
                    candidates = []
                    original_vector = self.getvec(atom, main_atom)

                    for a in circledump:
                        comparison_vector = self.getvec(a, main_atom)
                        if flex_angle > self.angle(original_vector, comparison_vector) > -flex_angle:
                            candidates.append([a, self.distance(a, atom)])
                    candidates.sort(key=lambda x: x[1])

                    for c in candidates:
                        if (not self.no_overlap(c[0], pocket, cheat = 1.0)[0]):
                            if (self.no_overlap(c[0], lining)[0]):
                                gotcha = True
                                messiah = c[0]
                                break

                    if (gotcha):
                        ret.append(messiah)
                        messiah.tempf = atom.tempf
                        messiah.radius = chargerad
                        messiah.host = main_atom
        return ret

    # builds a single residue from given atoms
    def build_single_res(self, atoms, covrad, tolerance):
        for a in atoms:
            for a2 in atoms:
                if a.idn != a2.idn:
                    if self.distance(a, a2) < (covrad[a.elemen.lower()] + covrad[a2.elemen.lower()] + tolerance):
                        a.bonds.append(a2)

    # builds residues based on a covalent radius dictionary.
    def build_res(self, atoms, covrad, tolerance, checkH):
        residues = {}
        h = False
        for atom in atoms:
            residues.update({int(atom.resseq): atom.myAA})
            if (checkH):
                if (atom.elemen.lower() == "h"):
                    h = True
            for atom2 in atom.myAA:
                if (atom.idn != atom2.idn):
                    if (self.distance(atom, atom2) < (covrad[atom.elemen.lower()] + covrad[atom2.elemen.lower()] + tolerance)):
                        if atom2.elemen.lower() == "h":
                            atom.bonds_to_h.append(atom2)
                        else:
                            atom.new_bonds.append(atom2)
                        atom.bonds.append(atom2)
            if (len(atom.bonds) < 1 and len(atom.myAA) > 1):
                dist_rank = []
                for a in atom.myAA:
                    if (a.idn != atom.idn):
                        dist_rank.append([a, self.distance(a, atom)])
                dist_rank.sort(key=lambda x: x[1])
                # TODO: implement a library fetch for this. jostain kirjastosta
                # oikeat määrät atomeja sidoksiksi.
                # lähimmät niin monta = sidokset.
                atom.bonds.append(dist_rank[0][0])
        return [h, residues]

    # returns the determinant of an array. Array must be in form [[],[],[]] and
    # only contain numbers.
    def array_det(self,l):
        n=len(l)
        if (n>2):
            i=1
            t=0
            current_sum=0
            while t<=n-1:
                d={}
                t1=1
                while t1<=n-1:
                    m=0
                    d[t1]=[]
                    while m<=n-1:
                        if (m != t):
                            d[t1].append(l[t1][m])
                        '''if (m==t):
                            u=0
                        else:
                            d[t1].append(l[t1][m])
                        '''
                        m+=1
                    t1+=1
                l1=[d[x] for x in d]
                current_sum=current_sum+i*(l[0][t])*(self.array_det(l1))
                i=i*(-1)
                t+=1
            return current_sum
        else:
            return (l[0][0]*l[1][1]-l[0][1]*l[1][0])


    # Picks all atoms connected to center atom. That is, secondary cavities
    # are eliminated.
    def remove_unconnected(self, center, pocket, bonds, angles, maxnumb):
        ret = []
        for atom in pocket:
            if (len(atom.bonds) > 2):
                ret.append(atom)
            elif (len(atom.bonds) == 2):
                atom1_vector = self.getvec(atom.bonds[0], atom) #[atom.bonds[0].x - atom.x, atom.bonds[0].y - atom.y, atom.bonds[0].z - atom.z]
                atom2_vector = self.getvec(atom.bonds[1], atom)  #[atom.bonds[1].x - atom.x, atom.bonds[1].y - atom.y, atom.bonds[1].z - atom.z]
                angle = self.angle(atom1_vector, atom2_vector)
                add = True
                for a in angles:
                    if ((a+1) > angle > (a-1)):
                        add = False
                if (not add and maxnumb > 0):
                    group1 = set([atom, atom.bonds[0]])
                    group2 = set([atom, atom.bonds[1]])
                    try:
                        self.follow_bonds(atom.bonds[0], group1)
                        self.follow_bonds(atom.bonds[1], group2)
                    except RuntimeError as e:
                        if "maximum recursion depth exceeded" in e.message:
                            self.fatal_recursion_error(pocket, center)
                        else:
                            self.print_this("fat-err", "Unknown error with follow_bonds!")
                            self.quit_now()
                    if (len(group1) < maxnumb or len(group2) < maxnumb or len(group1) == len(group2)):
                        add = True
                if (add):
                    ret.append(atom)
                else:
                    for at in atom.bonds:
                        # remove atom from all bonds
                        at.bonds = [x for x in at.bonds if x != atom]
            elif (len(atom.bonds) == 1):
                ret.append(atom)
        safe = set()
        try:
            self.follow_bonds(center, safe)
        except RuntimeError as e:
            if "maximum recursion depth exceeded" in e.message:
                self.fatal_recursion_error(pocket, center)
            else:
                self.print_this("fat-err", "Unknown error with follow_bonds!")
                self.quit_now()
        return list(safe)

    # error message and controlled crash after fatal recursion error.
    def fatal_recursion_error(self, pocket, center):
        self.print_this("fat-err", "Maximum recursion depth exceeded!")
        self.print_this("err", "This crash is due to input boxradius being far too large, or not enough atoms being eliminated in previous steps.")
        self.print_this("err", "Try decreasing box radius or checking center coordinates, for it is possible the box center is in the wrong place. This error should not occur with a radius < 15, and it should be exceedingly rare with smaller radiuses as well.")
        self.print_this("err", "Sometimes surface pockets may cause problems. If you have not done so yet, you may wish to check \"Keep_anyway_radius\" in the input file. Usually it takes care of any troubles with surface pockets, but a number too large there can also be problematic.")
        self.print_this("err", "Box is currently centered around (" + str(center.x) + ", " + str(center.y) + ", " + str(center.z) + "). Is this correct?")
        self.print_this("err", "If both radius and center are correct, then the cause is unknown. If necessary, contact author with a detailed description of the problem and all input files.")
        self.print_this("err", "Outputting >>>>>>>\"TaskuError.mol2\"<<<<<<<. Check the file to identify the problem.")
        self.print_this("err", "Number of atoms: " + len(pocket))
        self.print_this("fil", self.get_mol2_output(pocket, None, "ERROR", False), output=open("TaskuError.mol2", 'w'))
        self.print_this("err", "Exiting...")
        self.quit_now()

    # follows bonds on atoms, adding atoms to lastset on the way.
    def follow_bonds(self, startpoint, lastset, weak=False, limit_number=0):
        lastset.add(startpoint)
        for atom in startpoint.bonds:
            if atom not in lastset:
                if (weak):
                    if (not len(lastset) >= limit_number):
                        self.follow_bonds(atom, lastset, weak, limit_number)
                else:
                    self.follow_bonds(atom, lastset, weak, limit_number)

    # Picks all atoms connected to center atom. That is, secondary cavities are
    # eliminated.
    def onlyConnected(self, allatoms, center, cutoff, no=None):
        safe = self.getwithinrad(center, allatoms, cutoff)
        for safetom in safe:
            if no is not None:
                if safetom != no:
                    close_atoms = self.getwithinrad(safetom, allatoms, cutoff)
                    safetom.close = close_atoms
                safe.extend(close_atoms)

            else:
                close_atoms = self.getwithinrad(safetom, allatoms, cutoff)
                safetom.close = close_atoms
                safe.extend(close_atoms)
        return safe

    # adds bonds between atoms. bonds from atom a are added to all other atoms
    # that are within distance of a
    def add_bonds(self, fillers, distance):
        bonds_dict = {}
        for a in fillers:
            a.bonds = []
            for a2 in fillers:
                if (not a.idn == a2.idn):
                    a_to_a2_distance = self.distance(a, a2)
                    if (a_to_a2_distance <= distance):
                        if(a.idn < a2.idn):
                            bonds_dict[(a, a2)] = [a, a2, a_to_a2_distance]
                        else:
                            bonds_dict[(a2, a)] = [a2, a, a_to_a2_distance]
                        a.bonds.append(a2)
        #self.print_this("err", distance)
        return bonds_dict

    # Deletes atoms farther than a given distance from protein.
    def delete_farther_than(self, distance, pocket, protein):
        if (distance == 0):
            return [pocket, []]
        ret = []
        disq = []
        for a in pocket:
            add = False
            for p in protein:
                if (self.distance(a, p) < distance):
                    add = True
                    break
            if (add):
                ret.append(a)
            else:
                disq.append(a)

        return [ret, disq]

    # shoots some awesome rays. These (rays from center to every atom in
    # pocketlining), will be used as a description of the shape that pocket
    # filling atoms have to be in. Checking will ultimately be done by checking
    # distances. This could also be achieved with a "point inside an utterly
    # irregular polyhedron" algorithm, but for that to work one would have to
    # define the faces of the polyhedron, which would ultimately make it far
    # less efficient.
    def createBoundaries(self, pocketLining, cutoff, fillerradius, center):
        cluster = []
        for a in pocketLining:
            cluster.extend(self.createray(a, center, cutoff, fillerradius))
        return cluster

    # returns six atoms around the given atom.
    def get_rad_6(self, center, radius):
        one = Atom("null", 1, "O", "", "WAT", "", 1, "", center.x + radius, center.y, center.z,"", "0.000", "O.3", "0")
        two = Atom("null", 2, "O", "", "WAT", "", 1, "", center.x - radius, center.y, center.z,"", "0.000", "O.3", "0")
        three = Atom("null", 3, "O", "", "WAT", "", 1, "", center.x, center.y + radius, center.z,"", "0.000", "O.3", "0")
        four = Atom("null", 4, "O", "", "WAT", "", 1, "", center.x, center.y - radius, center.z,"", "0.000", "O.3", "0")
        five = Atom("null", 5, "O", "", "WAT", "", 1, "", center.x, center.y, center.z + radius,"", "0.000", "O.3", "0")
        six = Atom("null", 6, "O", "", "WAT", "", 1, "", center.x, center.y, center.z - radius,"", "0.000", "O.3", "0")
        return [one, two, three, four, five, six]

    # merely creates a cube of atoms. nothing else.
    def create_cube(self, center, radius, fillerradius, packing):
        filleratoms = []
        if packing == "cube":
            centers = center
            nextid = 2
            xrad = radius
            yrad = radius
            zrad = radius
            distancebetween = fillerradius * 2
            counter = 0
            # TODO: Possibly remove iteration over centers
            # TODO: Possibly accept only one center as parameter
            for center in centers:
                x = center.x
                y = center.y
                z = center.z
                i = x + float(xrad)
                while i > x - float(xrad):
                    j = y + float(yrad)
                    while j > y - float(yrad):
                        k = z + float(zrad)
                        counter += 1
                        while k > z - float(zrad):
                            newAtom = Atom("null", nextid, "O", "", "WAT", "", nextid, "", i, j, k,"", "0.000", "C", "0")
                            newAtom.radius = fillerradius
                            filleratoms.append(newAtom)
                            nextid = nextid + 1
                            k = k - distancebetween
                        j = j - distancebetween
                    i = i - distancebetween
        elif packing == "bcc":
            center = center[0]
            nextid = 0
            c = (2.0 / math.sqrt(3))
            dist_between = c * fillerradius * 2
            x = center.x
            y = center.y
            z = center.z
            radius = float(radius)
            i = x + radius
            j_stag = False
            k_stag = False
            while i > x - radius:
                j = y + radius
                if j_stag:
                    j += c * fillerradius
                    j_stag = False
                else:
                    j_stag = True
                if k_stag:
                    k_stag = False
                else:
                    k_stag = True
                while j > y - radius:
                    k = z + radius
                    if k_stag:
                        k += c * fillerradius
                    while k > z - radius:
                        newAtom = Atom("null", nextid, "O", "", "BOX", "", nextid, "", i, j, k, "", "0.000", "C", "0")
                        newAtom.radius = fillerradius
                        filleratoms.append(newAtom)
                        nextid += 1
                        k -= dist_between
                    j -= dist_between
                i -= c * fillerradius
        elif packing == "fcc":
            center = center[0]
            nextid = 0
            c = math.sqrt(2)
            dist_between = 2 * c * fillerradius
            x = center.x
            y = center.y
            z = center.z
            radius = float(radius)
            i = x + radius
            i_stag = 0
            while i > x - radius:
                j_stag = 0
                j = y + radius
                while j > y - radius:
                    k = z + radius
                    if i_stag % 2 == 1:
                        k += 0.5 * dist_between
                    if j_stag % 2 == 0:
                        k += 0.5 * dist_between
                    while k > z - radius:
                        newAtom = Atom("null", nextid, "O", "", "BOX", "", nextid, "", i, j, k, "", "0.000", "C", "0")
                        newAtom.radius = fillerradius
                        filleratoms.append(newAtom)
                        nextid += 1
                        k -= dist_between
                    j_stag += 1
                    j -= 0.5 * dist_between
                i_stag += 1
                i -= 0.5 * dist_between
        else:
            self.print_this("fat-err", "Packing method {} not recognized.".format(packing))
            self.quit_now()
        return filleratoms

    # returns True, if atom is charged.
    def is_charged(self, atom):
        if (atom.elemen.lower() == "c"):
            return False
        if (atom.elemen.lower() == "s"):
            return False
        if (atom.elemen.lower() == "h"):
            if (len(atom.bonds) < 0):
                return False
            else:
                if atom.bonds[0].elemen.lower() == "c":
                    return False
        if (atom.new_ch != 0.0):
            return True
        return False

    # Utilities

    # renumbers the atoms, starting with 1
    def number_and_name_these(self, retatoms, resname, mincharge):
        if self.dontname:
            return retatoms
        seqid = 1
        idnum = 1
        newret = []
        for l in retatoms:
            oidx = 1
            nidx = 1
            cidx = 1
            atoms = l[1]
            atoms.sort(key=lambda x: x.name)
            for atom in atoms:
                if (atom.tempf > mincharge):
                    elemen = "N"
                    nameline = "N"
                    idx = nidx
                    nidx += 1
                elif (atom.tempf < -mincharge):
                    elemen = "O"
                    nameline = "O"
                    idx = oidx
                    oidx += 1
                else:
                    elemen = "C"
                    nameline = "C"
                    idx = cidx
                    cidx += 1
                atom.elemen = elemen + ".3"
                idline = str(idx)
                if idx < 10000:
                    idline += " "
                if idx < 1000:
                    idline += " "
                if idx < 100:
                    idline += " "
                if idx < 10:
                    idline += " "
                atom.name = nameline + idline
                if (atom.host is not None):
                    atom.elemen == atom.host.elemen
                atom.resname = resname
                atom.resseq = seqid
                atom.idn = idnum
                idnum += 1
            atoms.sort(key=lambda x: x.idn)
            seqid += 1
            newret.append([l[0], atoms, l[2]])
        return newret

    # print function. prints to error output on default, prepends message
    # type(depending on ar) to the line before printing.
    def print_this(self, ar,  line, output=sys.stdout):
        thor = ar.split("-")
        if (type(line) is not str and not ar=="fil"):
            line = str(line)
        if type(line) is str:
            lines = line.split("\n")
        printline = ""
        if (ar == "err"):
            output=sys.stderr
            for l in lines:
                printline  += self.color["PURPLE"] + "ERROR: " + self.color["END"] + l + "\n"
        elif (ar == "fat-err"):
            output=sys.stderr
            printline = self.color["RED"] + "# # # # # # # # " + "\n"
            for l in lines:
                printline  += "FATAL ERROR: " + l + "\n"
            printline  +=  "# # # # # # # # " + self.color["END"] + "\n"
        elif (ar == "inf"):
            for l in lines:
                printline  +=  "INFO: " + l + "\n"
        elif (ar == "non"):
            for l in lines:
                printline  +=  l + "\n"
        elif (ar == "war"):
            output=sys.stderr
            for l in lines:
                printline  +=  self.color["YELLOW"] + "WARNING: " + self.color["END"] + l + "\n"
        elif (thor[0] == "ver"):
            if (int(thor[1]) <= self.verbose):
                for l in lines:
                    printline += "VERBOSE: " + l + "\n"
        elif (ar == "fil"):
            printline = ""
            if type(line) is list:
                for a in line:
                    printline += a
            else:
                printline = line
        if len(printline) > 0:
            # [:-1] for removing the last line break
            print(printline[:-1])

    # returns an average point of the given atomgroup
    def get_avg_point(self, atomgroup):
        #~ avg = [atomgroup[0].x,atomgroup[0].y,atomgroup[0].z]
        x_tally = 0.0
        y_tally = 0.0
        z_tally = 0.0
        for i, atm in enumerate(atomgroup):
            x_tally += atm.x
            y_tally += atm.y
            z_tally += atm.z
        avg = [x_tally/len(atomgroup), y_tally/len(atomgroup), z_tally/len(atomgroup)]
            #~ avg = [(avg[0]+atomgroup[i].x)/2, (avg[1]+atomgroup[i].y)/2, (avg[2]+atomgroup[i].z)/2]
        return Atom("null", 0, "O", "", "WAT", "", 0, "", avg[0], avg[1], avg[2],"", "0.000", "O.3", "0")

    # Atom radius settings

    # Parses and applies level 1 (atom type specific) radii to atoms.
    def runlevel1(self, atoms, level1data):
        rAllOthers = 10.0
        for line in level1data:
            if ("AllOthers" in line[0]):
                rAllOthers = float(line[1])
        for a in atoms:
            found = False
            for line in level1data:
                if (not found and line[0] in a.name.lower()):
                    a.radius = float(line[1])
                    found = True
            if (not found):
                a.radius = rAllOthers

    # Returns the first letter of a string. Not number. Letter.
    def get_element(self, stringamaba):
        elemen = False
        for c in stringamaba:
            if c == "\"":
                elemen = True
            if c.isalpha():
                return (c, elemen)


    # Parses, sorts and applies level 2 (amino acid) radii to atoms
    def runlevel2(self, atoms, level2data):
        lvl2dict = defaultdict(list)
        append_dict = {}
        for line in level2data:
            lvl2dict[line[0] + line[1] + line[4]] = line
            elmdata = self.get_element(line[4]).lower()
            elmstr = elmdata[0]
            if elmdata[1]:
                elmstr = "e"+elmstr
            append_dict.update({line[0]+line[1]+elmstr: line})

        for atom in atoms:
            do = False


            if ((atom.resname.lower() + "000" + "000") in lvl2dict):
                data = lvl2dict[atom.resname.lower() + "000" + "000"]
                do = True

            if ((atom.resname.lower() + str(atom.resseq) + "e" + atom.elemen.lower()) in append_dict):
                data = append_dict[atom.resname.lower() + str(atom.resseq) + "e" + atom.elemen.lower()]
                do = True
            elif ((atom.resname.lower() + "000" + "e" + atom.elemen.lower()) in append_dict):
                data = append_dict[atom.resname.lower() + "000" + "e" + atom.elemen.lower()]
                do = True

            if ((atom.resname.lower() + str(atom.resseq) + atom.name.lower()) in lvl2dict):
                data = lvl2dict[atom.resname.lower() + str(atom.resseq) + atom.name.lower()]
                do = True
            elif ((atom.resname.lower() + "000" + atom.name.lower()) in lvl2dict):
                data = lvl2dict[atom.resname.lower() + "000" + atom.name.lower()]
                do = True

            if (do):
                if "1" == data[2]:
                    atom.radius = atom.radius + float(data[3])
                else:
                    if (float(data[3]) <= 0):
                        self.print_this("err", "Radius of 0 or below specified for " + atom.resname + " " + atom.name + "! IGNORING")
                    else:
                        atom.radius = float(data[3])

    # Parses and applies level 3 (sequence ID) radius data to atoms.
    def runlevel3(self, atoms, level3data):
        lvl3dict = defaultdict(list)
        for line in level3data:
            lvl3dict[int(line[0])] = line
        for atom in atoms:
            if atom.idn in lvl3dict:
                data = lvl3dict[atom.idn]
                if "1" == data[1]:
                    atom.radius = atom.radius + float(data[2])
                else:
                    if (float(data[2]) <= 0):
                        self.print_this("err", "Radius of 0 or below specified! IGNORING")
                    else:
                        atom.radius = float(data[2])

    # Administers the radius-parsing process.
    def setradi(self, proteinatoms, distancedata):
        level1radii = []
        level2radii = []
        level3radii = []
        section2 = False
        section3 = False
        distancedata = [_f for _f in distancedata if _f]
        for i in range(0, len(distancedata)):
            if (section3):
                level3radii.append(distancedata[i].lower().split())
            if (section2 and not section3):
                if (":::-:::" in distancedata[i]):
                    section3 = True
                else:
                    level2radii.append(distancedata[i].lower().split())
            if (not section2):
                if (":::-:::" in distancedata[i]):
                    section2 = True
                else:
                    level1radii.append(distancedata[i].lower().split())
        level1radii.sort(self.atomfieldkey)
        level2radii.sort(key=lambda x: x[1], reverse=True)
        self.runlevel1(proteinatoms, level1radii)
        if (len(level2radii) > 0):
            self.runlevel2(proteinatoms, level2radii)
        if (len(level3radii) > 0):
            self.runlevel3(proteinatoms, level3radii)

    # returns [True, number] if number (a string) contains only an int.
    # otherwise returns [false]
    def is_int(self, number):
        try:
            d = int(number)
            return [True, d]
        except ValueError:
            return [False]

    # returns atoms with the given seqID
    def get_AA(self, seqID, protein, chain=None):
        ret = []
        val = False
        for p in protein:
            if (int(p.resseq) == int(seqID)):
                val = True
                ret.append(p)
            if (val and int(p.resseq) != int(seqID)):
                return ret
        return ret

    # returns atoms with the given seqID
    def get_AAs(self, seqID_set, protein, chain=False):
        ret = []
        if not chain:
            for p in protein:
                if (int(p.resseq) in seqID_set):
                    ret.append(p)
        else:
            for entry in seqID_set:
                line = entry.split("-")
                for p in protein:
                    if (str(p.resseq) == line[1]):
                        if (p.chainid.lower() == line[0]):
                            ret.append(p)
        return ret

    # Atom list manipulations

    # deletes atoms based on information from angledata.
    def angleBlast(self, fillers, allatoms, center, angledata):
        endpoint = None
        legal = set([])
        for line in angledata:
            #Parsing data:
            #distance placement for the angle
            dist = line[2]
            if ("+" in dist):
                dist = float(dist[1:])
            elif("-" in dist):
                dist = -float(dist[1:])
            else:
                dist = 0
            neg = False
            delete_angle = float(line[1])
            # if angle is too small - do nothing
            if (-1.0 < delete_angle < 1.0 ):
                return fillers
            if (delete_angle < 0):
                # negative angle = angle in opposite direction. thus: 180-angle
                # and the actual "angle" is the area not covered by this.
                delete_angle = 180 - delete_angle
                neg = True
            if("aa" == line[0][:2]):
                endpoint = self.get_avg_point(self.get_AA(line[0][2:], allatoms))
            elif("a" == line[0][:1]):
                endpoint = self.getAtom(line[0][1:], allatoms)

            #Blastoff
            if endpoint:
                #Few more parameters. These are optional, so may be present or
                # not.
                opening = False
                if (len(line) > 3):
                    startpoint = self.getAtom(line[3], allatoms)
                if (len(line) > 4 and line[3] == "o"):
                     # opening = specified by "o"
                    opening = True
                #new startpoint if distance != 0
                if (dist != 0):
                    r = dist/self.distance(startpoint, endpoint)
                    x = ((startpoint.x - endpoint.x) * r) + startpoint.x
                    y = ((startpoint.y - endpoint.y) * r) + startpoint.y
                    z = ((startpoint.z - endpoint.z) * r) + startpoint.z
                    startpoint = Atom("null", 0, "Cl", "", "FIL", "", "0", "", x, y, z,"", "0.000", "Cl", "0")

                center_vec = self.getvec(endpoint, startpoint) # [endpoint.x - startpoint.x, endpoint.y - startpoint.y, endpoint.z - startpoint.z]
                protangles = []
                # if opening is not True, none of this info will be used. If
                # opening is True, it'll be rather faster if we have it ready,
                # than if we were to check every atom every time.
                if(opening):
                    for a in allatoms:
                        vec_c_to_a = self.getvec(a, startpoint) #[a.x - startpoint.x, a.y - startpoint.y, a.z - startpoint.z]
                        if (not sum(vec_c_to_a) == 0):
                            protangles.append([a, self.angle(center_vec, vec_c_to_a), self.distance(startpoint, a)])

                # And now for the actual blasting...
                for atom in fillers:
                    vec_c_to_atom = self.getvec(atom, startpoint) #[atom.x - startpoint.x, atom.y - startpoint.y, atom.z - startpoint.z]
                    current_angle = self.angle(center_vec, vec_c_to_atom)
                    add = True
                    # if negative angle has been given, the investigated angle
                    # should be bigger to qualify for deletion
                    if (neg and current_angle > delete_angle):
                        add = False
                    # otherwise it should be smaller.
                    if (not neg and current_angle < delete_angle):
                        add = False
                    # let's investigate if the atom should be added anyway, in
                    # case opening -switch is used.
                    # TODO: not implemented
                    #if (opening and not add):

                    if(add):
                        legal.add(atom)
        return list(legal)

    # returns the dot product of two vectors.
    def dotproduct(self, v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

    # returns length of a given vector
    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    # cleans cos.
    def clean_cos(self,cos_angle):
        return min(1,max(cos_angle,-0.99999))

    # returns angle between two vectors.
    def angle(self, v1, v2, radians=False):
        lenlen = self.length(v1) * self.length(v2)
        #TODO: definitely not a pretty solution. need a better one.
        dop = self.clean_cos(self.dotproduct(v1, v2) / lenlen)
        ret = math.acos(dop)
        if not radians:
            ret = math.degrees(ret)
        return ret

    # returns the atom closest to oldcenter that does not overlap with protein
    # atoms.
    def newcenter(self, oldcenter, setti, protein):
        distanceinfo = []
        for a in setti:
            distanceinfo.append(a)
            a.distancetocenter = self.distance(a, oldcenter)
        distanceinfo.sort(self.centerkey, reverse = True)
        for atom in distanceinfo:
            if (self.no_overlap(atom, protein)[0]):
                atom.name = "O"
                atom.elemen = "O"
                return atom
        return distanceinfo[0]

    # adds hetatoms whose resname can be found in hetatmaccountedfor to list of
    # protein atoms.
    def get_resname(self, allatoms, hetatmaccountedfor):
        ret = []
        for heta in allatoms:
            for entry in hetatmaccountedfor:
                if (entry.lower().strip() == "all" or entry.strip().lower() == heta.resname.lower().strip()):
                    ret.append(heta)
        return ret

    # returns atom of a given ID
    def getAtom(self, idNumber, atoms):
        for a in atoms:
            if (a.idn == int(idNumber)):
                return a
        return None

    # returns all atoms from atomlist that do not overlap with proteinatoms
    def nonOverlapping(self, proteinatoms, atomlist, circle_cheat={}, fillmetal=[False], halp=""):

        if (len(proteinatoms) < 1):
            return [atomlist, []]
        returnlist = []
        disqualified = []
        fill_res = None
        if (fillmetal[0]):
            fill_res = set()
            for metal_atom in fillmetal[1]:
                fill_res.add(metal_atom.resseq)
        #~ counter = 0
        for filler in atomlist:
            add = False
            if (self.no_overlap(filler, proteinatoms, circle_cheat=circle_cheat, emptyres=fill_res)[0]):
                add = True
            if (add):
                returnlist.append(filler)
            else:
                disqualified.append(filler)
        return [returnlist, disqualified]

    # only returns atoms from atoms that overlap with atoms in criteria list
    def overlappingOnly(self, atoms, criteria, protein, radcheat=0.0, keep_anyway=None):
        ret = set()
        #dist = []
        try_again = []
        number_included = 0
        for a in atoms:
            ov = self.no_overlap(a, criteria, cheat=radcheat)
            if(not ov[0]):
                number_included = number_included + 1
                ret.add(a)
            else:
                try_again.append([a, ov[1], ov[2]])
        second_round = []
        disqualified = set()
        ret = list(ret)


        if (keep_anyway is not None):
            disqualified = set()
            for aline in try_again:
                kep = self.get_within_radius(aline[0], keep_anyway[2], keep_anyway[0])
                add = False
                if (keep_anyway[1] > 0):
                    AAs = set()
                    for ak in kep:
                        AAs.add(ak.resseq)
                    if (len(AAs) >= keep_anyway[1]):
                        add = True
                elif (aline[2][1] < keep_anyway[0]):
                    add = True
                else:
                    kep = self.get_within_radius(aline[0], keep_anyway[3], keep_anyway[0])
                    if (keep_anyway[1] > 0):
                        AAs = set()
                        for ak in kep:
                            AAs.add(ak.resseq)
                        if (len(AAs) >= keep_anyway[1]):
                            add = True
                    elif (aline[2][1] > keep_anyway[0]):
                        add = True
                if (add):
                    second_round.append(aline[0])
                else:
                    disqualified.add(aline[0])
            if (not self.is_null(keep_anyway[4])):
                # first remove all whitespace and then split at ;
                keepdata = "".join(keep_anyway[4].split()).split(";")
                #distdata = []
                compl = False
                com_criteria = []
                com_distdata = {}
                retu = []
                disqualified = set()
                for line in keepdata:
                    if "+" not in line:
                        criteria_entry = line.split("-")
                        more = self.get_within_radius_of_set(float(criteria_entry[0]), [second_round], self.get_AA(criteria_entry[0], protein))
                        retu.extend(more)
                    else:
                        crit_entry = line.split("+")
                        com_criteria.append(crit_entry)
                        compl = True

                        for c in crit_entry:
                            real_entry = c.split("-")
                            com_distdata.update({real_entry[0]: self.get_within_radius_of_set(float(real_entry[1]), [second_round], self.get_AA(real_entry[0], protein))})
                if (compl):
                    setret = set()
                    for atom in second_round:
                        for line in com_criteria:
                            add = True
                            for entry in line:
                                if atom not in com_distdata[entry.split("-")[0]]:
                                    add = False
                            if add:
                                setret.add(atom)
                            else:
                                disqualified.add(atom)
                    retu.extend(list(setret))
                ret.extend(retu)

            else: ret.extend(second_round)
        else:
            for a in try_again:
                disqualified.add(a[0])
        return [ret, list(disqualified)]

    # Various distance tools:

    #gets two atoms that are farthest apart from one another in atomlist
    def get_farthest_apart(self, atomlist):
        distanceinfo = []
        for a in atomlist:
            for b in atomlist:
                if (not a.idn == b.idn):
                    distanceinfo.append([a, b, self.distance(a, b)])
        # Delete empty lists from distanceinfo.
        distanceinfofin = [x for x in distanceinfo if x != []]
        distanceinfofin.sort(key= lambda x: x[2], reverse=True)

        return [distanceinfofin[0][0], distanceinfofin[0][1]]

    #gets two atoms that are farthest apart from one another in atoms
    def get_farthest_apart_basic(self, atoms):
        far = (atoms[0], atoms[0], 0.0)
        for i in range(0, len(atoms)):
            hurr = atoms[i:]
            for j in range(0, len(hurr)):
                dist = self.distance(atoms[i], hurr[j])
                if (dist > far[2]):
                    far = [atoms[i], hurr[j], dist]
        return far

    # gets the atom in candidates that is farthest from atom
    def get_farthest_from(self, atom, atomlist):
        dist = [atomlist[0], self.distance(atomlist[0], atom), 0]
        for i, a in enumerate(atomlist):
            d = self.distance(atom, a)
            if (d > dist[1]):
                dist = [a, d, i]
        return dist

    # gets the atom in candidates that is farthest from the avg of atoms from
    # fromlist
    def get_farthest_from_avg(self, candidates, fromlist):
        dist = [None, 0]
        for a in candidates:
            avg = 0
            for b in fromlist:
                avg = avg + self.distance(a, b)
            avg = avg / len(fromlist)
            if (avg > dist[1]):
                dist = [a, avg]
        return dist

    # returns all atoms from allatoms that are within a given radius of atoms
    def withinRad(self, atoms, allatoms, radius):
        ret = []
        for a in allatoms:
            for a2 in atoms:
                if self.distance(a, a2) < radius:
                    ret.append(a)
                    break
        # removes duplicates
        return list(set(ret))

    # Checks for overlap based on radius of a given atoms
    def no_overlap(self, atom, others, circle_cheat={}, cheat=0.0, emptyres=None):
        closedata = {}
        closest = [None, 10000]
        same_res_distance = self.same_residue_distance_tolerance
        for atom2 in others:
            if (emptyres is not None):
                if (atom2.resseq in emptyres):
                    continue
            atom2rad = atom2.radius
            dist = self.distance(atom, atom2)
            key = (atom2.resname + str(atom2.resseq))
            if (key in closedata):
                if (closedata[key] > dist):
                    closedata[key] = dist
            else:
                closedata.update({key: dist})
            if (dist < closest[1]):
                closest = [atom2, dist]
            if dist <= (atom.radius + atom2rad):
                if (len(circle_cheat) > 0):
                    if (atom2.elemen.lower() in circle_cheat):
                        if (dist <= atom.radius + circle_cheat[atom2.elemen.lower()]):
                            return [False, atom2]
                    else:
                        return [False, atom2]
                elif atom.host is not None:
                    if atom.host.resseq == atom2.resseq:
                        if dist < same_res_distance:
                            return [False, atom2]
                    else:
                        return [False, atom2]
                else:
                    return [False, atom2]
            if cheat != 0.0:
                if dist <= cheat:
                    return [False, atom2]
        return [True, closedata, closest]

    # Returns all atoms within a given radius
    def get_within_radius(self, atom, allatoms, cutoff, addbond = False):
        retlist = []
        for second in allatoms:
            if (0 < self.distance(atom, second) <= cutoff):
                retlist.append(second)
                if (addbond):
                    second.bonds.append(atom)
        return retlist

    # Returns all atoms within a given radius
    def getwithinrad(self, atom, allatoms, cutoff):
        retlist = []
        for second in allatoms:
            if (not second.safe):
                if (self.distance(atom, second) <= cutoff):
                    second.safe = True
                    retlist.append(second)
        return retlist

    # calculates the distance between two atoms in three-dimensional space
    def distance(self, a, b):
        xd = a.x-b.x
        yd = a.y-b.y
        zd = a.z-b.z
        distance = math.sqrt(xd*xd + yd*yd + zd*zd)
        return distance

    # returns atom closest to atom in atomlist. Assumes that atom IS NOT FOUND
    # in atomlist.
    def get_closest(self, atom, atomlist):
        best = [atomlist[0], self.distance(atom, atomlist[0]), -1, None]
        ranking = []
        for i in range(0, len(atomlist)):
            if atom != atomlist[i]:
                dist = self.distance(atom, atomlist[i])
                ranking.append([atomlist[i], dist, i])
                #~ if (dist < best[1]):
                    #~ best[3] = best[0]
                    #~ best[0] = atomlist[i]
                    #~ best[1] = dist
                    #~ best[2] = i
        ranking.sort(key=lambda x: x[1])
        if len(ranking) > 1:
            best = ranking[0]
            best.append(ranking[1][0])

        return best

    # File tools

    # for parsing atoms from a .pdb file.
    def parse_pdb(self, filename, delchain, charge_lib, pdb_charge, sign_charge):
        self.print_this("ver-1", "Parsing .pdb: " + filename)
        prot = []
        het = []
        allA = []
        lines = self.read(filename)
        chains = {}
        hetchains = {}
        amino_acids = {}
        ligs = {}
        water_only = set()
        del_list = []
        nocharge = []
        # TODO: atoms from angle library.
        chset = set(["fe", "cu", "co", "mn", "mg", "ca", "zn", "o", "n"])
        biggest_resnum = 0
        for idx, line in enumerate(lines):
            #lastatom = None
            # Only if the line starts with "ATOM" or "HETATM",
            # it will be read and atom info saved.
            if not ("ATOM" in line[:6] or "HETATM" in line[:6]):
                continue
            #initialize values, so each will have at least SOME value
            renam=name=altloc=resname=chainid=icode=occup=elemen=""
            serial=resseq=0
            x=y=z=tempf=0.0
            # the field locations are from .pdb specifications:
            # http://www.wwpdb.org/documentation/format33/sect9.html
            renam = line[:6].strip()
            serial = line[6:11].strip()
            name = line[12:16].strip()
            altloc = line[16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            x = line[30:38].strip()
            y = line[38:46].strip()
            z = line[46:54].strip()
            occup = line[54:60].strip()
            tempf = line[60:66].strip()
            elemen = line[76:78].strip()
            charge = line[78:].split()

            if int(resseq) > biggest_resnum:
                biggest_resnum = int(resseq)


            if len(charge) > 0:
                charge = charge[0].strip()
            else:
                charge = ""
            if resname.lower().strip() == "hem" and ("f" in elemen.lower() or "e" in elemen.lower()):
                elemen = "FE"

            if len(elemen) < 2 and len(name) > 1:
                newel = line[76:79].strip()
                if (len(newel) > 2):
                    newel = newel[:2]
                if newel in name:
                    elemen = newel

            try:
                tempfloat = float(tempf)
            except ValueError:
                splitchars = ["+", "-"]
                splitidx = 0
                for idx, char in enumerate(tempf):
                    if char in splitchars:
                        splitidx = idx
                try:
                    tempfloat = float(tempf[splitidx:])
                except ValueError:
                    self.print_this("err", "Error parsing a charge!")
                    self.print_this("err", "Atom number: " + serial)
                    self.print_this("err", "Problem field: temperature factor")
                    self.print_this("err", "Problem field contents: " + tempf)
                    self.print_this("err", "Ignoring atom. This may lead to further problems.")
                    continue

            newAtom = Atom(renam, serial, name, altloc, resname, chainid, resseq, icode, x, y, z, occup, tempfloat, elemen, charge)

            if (pdb_charge):
                newAtom.new_ch = newAtom.tempf
            else:
                if (("+" in tempf or "-" in tempf) and sign_charge):
                    newAtom.new_ch = tempfloat
                elif (newAtom.resname.lower() in charge_lib):
                    if (newAtom.name.lower() in charge_lib[newAtom.resname.lower()]):
                        newAtom.new_ch = charge_lib[newAtom.resname.lower()][newAtom.name.lower()]
                    else:
                        found = False
                        for syn_list in charge_lib["synonyms"]:
                            if newAtom.resname.lower() in syn_list:
                                for nam in syn_list:
                                    if nam in charge_lib:
                                        if newAtom.name.lower() in charge_lib[nam]:
                                            newAtom.new_ch = charge_lib[nam][newAtom.name.lower()]
                                            found = True
                                        elif newAtom.elemen.lower() in chset and not newAtom.resname.lower() == "pro":
                                            # TODO: add exceptions from the
                                            # angles.lib exclude list
                                            nocharge.append(newAtom.resname+str(newAtom.resseq) + ": " + newAtom.name)

                        if not found: # unreachable code for now.
                            for cand_ele in list(charge_lib[newAtom.resname.lower()].keys()):
                                if newAtom.elemen in cand_ele:
                                    newAtom.new_ch = charge_lib[newAtom.resname.lower()][cand_ele]
                                    found = True
                                    break
                            if not found:
                                for syn_list in charge_lib["synonyms"]:
                                    if newAtom.resname.lower() in syn_list:
                                        for nam in syn_list:
                                            if nam in charge_lib:
                                                for atom_entry in list(charge_lib[nam].keys()):
                                                    if newAtom.elemen.lower() in atom_entry:
                                                        newAtom.new_ch = charge_lib[nam][atom_entry]
                                                        found = True
                                                        break
            if (len(newAtom.elemen) < 1):
                cand = re.findall("\D", name)
                el_str = ""
                for i in range(0, len(cand)):
                    if cand[i].isupper():
                        el_str += cand[i]
                        if (len(cand) > (i + 1) and not cand[i + 1].isupper()):
                            el_str += cand[i+1]
                        break
                newAtom.elemen = ''.join(e for e in el_str if e.isalpha())

            if ("ATOM" in newAtom.renam):
                if (delchain is not None and newAtom.chainid.lower() == delchain):
                    del_list.append(newAtom)
                else:
                    if (newAtom.chainid + "-" + str(newAtom.resseq)) in amino_acids:
                        amino_acids[newAtom.chainid + "-" + str(newAtom.resseq)].append(newAtom)
                    else:
                        amino_acids[newAtom.chainid + "-" + str(newAtom.resseq)] = [newAtom]
                    prot.append(newAtom)
                    cid = newAtom.chainid.lower().strip()
                    if cid in chains:
                        chains[cid].append(newAtom)
                    else:
                        chains[cid] = [newAtom]
            else:
                if ("hoh" in newAtom.resname.lower() or "wat" in newAtom.resname.lower()):
                    water_only.add(newAtom)
                else:
                    het.append(newAtom)
                if newAtom.chainid + "-" + str(newAtom.resseq) in ligs:
                    ligs[newAtom.chainid + "-" + str(newAtom.resseq)].append(newAtom)
                else:
                    ligs[newAtom.chainid + "-" + str(newAtom.resseq)] = [newAtom]
                if newAtom.chainid in hetchains:
                    hetchains[newAtom.chainid].append(newAtom)
                else:
                    hetchains[newAtom.chainid] = [newAtom]
            allA.append(newAtom)
        new_amin = {}
        residues = []
        ligands = {}
        if len(nocharge) > 0:
            for a in nocharge:
                self.print_this("war", "No charge found for: " + a)
        i = 0
        for key, value in amino_acids.items():
            new_amin.update({i: value})
        #    residues.append(Residue(value))
            i = i + 1
        i = 0
        for key, value in ligs.items():
            ligands.update({i: value})
            i = i + 1
        for atom in prot:
            atom.myAA = amino_acids[atom.chainid + "-" + str(atom.resseq)]
        for atom in het:
            atom.myAA = ligs[atom.chainid + "-" + str(atom.resseq)]
        self.start_own_resnums = biggest_resnum + 1
        return [prot, het, allA, chains, hetchains, new_amin, ligands, amino_acids, water_only, del_list, ligs, residues]

    #builds mol2 file strings. prints out bonds if debug=True
    def get_mol2_output(self, mods, debug):
        idnum = 1
        returnstring = ""
        name_idx = 1
        if self.debug:
            dbmod = ["DEBUG", list(self.debug_atoms), []]
            mods.append(dbmod)
        for mod in mods:
            atomtable = []
            name = mod[0]
            atoms = mod[1]
            bonds = mod[2]
            for latom in atoms:
                name_idx_str = ""
                if self.debug:
                    name_idx_str += str(name_idx)
                x = "%.4f" % latom.x
                y = "%.4f" % latom.y
                z = "%.4f" % latom.z
                atomtable.append( [" " + str(idnum).rjust(6), " " + latom.name.rjust(2) + name_idx_str, " " + x.rjust(16), " " + y.rjust(9),
                                    " " + z.rjust(9), " " + latom.elemen.ljust(6), " " + str(latom.resseq).rjust(5),
                                    " " + (latom.resname + str(latom.resseq)).rjust(10), " " + "{:7.4f}".format(latom.tempf).rjust(8)] )
                idnum += 1
                name_idx += 1
            molstring = ""
            molstring += "@<TRIPOS>MOLECULE \n " + name + "\n " + str(len(atomtable)) + " \n SMALL \n USER_CHARGES \n \n@<TRIPOS>ATOM \n"
            for atomentry in atomtable:
                for datstr in atomentry:
                    molstring += datstr
                molstring += "\n"
            if (debug):
                self.print_this("inf", "debug output")
                molstring += "@<TRIPOS>BOND \n"
                i = 0
                for bond in bonds:
                    molstring += "   " + str(i) + "   " + str(bond[0].idn) + "   " +  str(bond[1].idn) + "   1\n"
                    i = i + 1
            molstring += "\n"
            molstring += "\n"
            returnstring += molstring
        return returnstring

    # removes comments from a list of strings (file lines)
    def remove_comments(self, linelist, args = None):
        noclines = []
        for line in linelist:
            if (len(line) > 2):
                # removes lines beginning with "#"
                if (line.find("#") == -1 or line.find("#") > 1):
                    # deletes parts after "#", as well as possible whitespace
                    app = line.split("#", 1)[0].strip()
                    if (args is not None):
                        # split at args[0], and take the number args[1] token.
                        app = app.split(args[0])[args[1]].strip()
                    noclines.append(app)
        return noclines

    # reads a file. Returns a list of lines without line breaks.
    def read(self, filename):
        try:
            f = open(filename.strip())
        except:
            self.print_this("fat-err", "Problem opening file!")
            self.print_this("err", filename.strip())
            self.print_this("err", "Make sure the file exists, can be read and whatnot.")
            self.quit_now()
        ret = []
        line = f.readline()
        while line:
            ret.append(line.strip())
            line = f.readline()
        f.close()
        return ret

    # Error :)
    def input_required(self):
        self.print_this("non", "_________ _        _______          _________")
        self.print_this("non", "\\__   __/( (    /|(  ____ )|\\     /|\\__   __/")
        self.print_this("non", "   ) (   |  \\  ( || (    )|| )   ( |   ) (   ")
        self.print_this("non", "   | |   |   \\ | || (____)|| |   | |   | |   ")
        self.print_this("non", "   | |   | (\\ \\) ||  _____)| |   | |   | |   ")
        self.print_this("non", "   | |   | | \\   || (      | |   | |   | |   ")
        self.print_this("non", "___) (___| )  \\  || )      | (___) |   | |   ")
        self.print_this("non", "\_______/|/    )_)|/       (_______)   )_(   ")
        self.print_this("non", "")
        self.print_this("non", " _______ _________ _        _______ ")
        self.print_this("non", "(  ____ \\\\__   __/( \\      (  ____ \\")
        self.print_this("non", "| (    \\/   ) (   | (      | (    \\/")
        self.print_this("non", "| (__       | |   | |      | (__    ")
        self.print_this("non", "|  __)      | |   | |      |  __)   ")
        self.print_this("non", "| (         | |   | |      | (      ")
        self.print_this("non", "| )      ___) (___| (____/\\| (____/\\")
        self.print_this("non", "|/       \\_______/(_______/(_______/")
        self.print_this("non", "")
        self.print_this("non", " _______  _______  _______          _________ _______  _______  ______   _" )
        self.print_this("non", "(  ____ )(  ____ \\(  ___  )|\\     /|\\__   __/(  ____ )(  ____ \\(  __  \\ ( )")
        self.print_this("non", "| (    )|| (    \\/| (   ) || )   ( |   ) (   | (    )|| (    \\/| (  \\  )| |")
        self.print_this("non", "| (____)|| (__    | |   | || |   | |   | |   | (____)|| (__    | |   ) || |")
        self.print_this("non", "|     __)|  __)   | |   | || |   | |   | |   |     __)|  __)   | |   | || |")
        self.print_this("non", "| (\\ (   | (      | | /\\| || |   | |   | |   | (\\ (   | (      | |   ) |(_)")
        self.print_this("non", "| ) \\ \\__| (____/\\| (_\\ \\ || (___) |___) (___| ) \\ \\__| (____/\\| (__/  ) _" )
        self.print_this("non", "|/   \\__/(_______/(____\\/_)(_______)\\_______/|/   \\__/(_______/(______/ (_)")
        self.print_this("non", "(-h for help)")

    # Input file generation, if requested:

    # help printing
    def print_help(self, altvalues):
        for a in altvalues:
            self.print_this("inf", a[1].ljust(14, " ") + ": " + a[4])
        self.print_this("inf", "")
        self.print_this("inf", "---")
        self.print_this("inf", "")
        self.print_this("inf", "Usage: python panther.py [options/args] inputfile outputfile")
        self.print_this("inf", "Options: ")
        self.print_this("inf", "-v for verbose")
        self.print_this("inf", "-h will print this help message.")
        self.print_this("inf", "-co to get center atoms only.")
        self.print_this("inf", "-bo for boundaries only.")
        self.print_this("inf", "-lo for lining atoms only.")
        self.print_this("inf", "-cho for charged atoms only.")
        self.print_this("inf", "-cbo for a cube of atoms only (the cube will be the set from which atoms will be eliminated during the run).")
        #self.print_this("inf", "-no for basic model only, no optional ones.")
        #self.print_this("inf", "-coor [filename] will override coordinate library setting in the input file.")
        #self.print_this("inf", "-rad [filename] will override radius library setting in the input file.")
        self.print_this("inf", "--automode enables automode. If this is enabled, centers possibly entered by user will be ignored and taskutus will try to find the cavity on its own. This relies on available ligand information, thus should not be used with bare .pdb files! ")
        self.print_this("inf", "")
        self.print_this("inf", "Additional switches (explained below):: --pdbch ; --signch ; -defval ; -mkdef <filename>")
        self.print_this("inf", "")
        self.print_this("inf", "--pdbch Take all charges from the .pdb file's TEMPF (temperature factor) field instead of the charge library.")
        self.print_this("inf", "--signch Take charges from the .pdb TEMPF only for atoms that have + or - sign in TEMPF. For those atoms lacking the + or - sign, charge is taken from the charge library.")
        self.print_this("inf", "")
        self.print_this("inf", "Everything else should be in input files. Three library files and one parameter file are required in addition to the .pdb. ")
        self.print_this("inf", "NOTE: all settings in inputfile can be defined on the command line. Command line arguments will OVERRIDE the input file. See above for available switches. ")
        self.print_this("inf", "Arguments with more than one word should be surrounded by \"\". i.e. -mbox Y -ofed \"Y Y\" ")
        self.print_this("inf", "-defval switch can be used to load default values. If this is used, there is no need for an input file. Parameters may then only be changed via the command line. ")
        self.print_this("inf", "-mkdef [filename] will output the default values and save it to the given file.")

    # returns a default file from altvalues
    def get_default_file(self, altvalues):
        ret = []
        component_lines = [["{}-{} ({}):: {}".format(a[0]+1, a[4], a[1], a[5]), a[6]] for a in altvalues]
        setting_len = max(list(map(len, [x[0] for x in component_lines]))) + 1
        wrapper = TextWrapper(width=160, subsequent_indent=' ' * setting_len + "# ")
        for line in component_lines:
            wrapped = wrapper.wrap("{:{width}s}{}\n".format(line[0], line[1], width=setting_len))
            ret.append("\n".join(wrapped) + "\n")
            #ret.append("{:{width}s}{}\n".format(line[0], line[1], width=setting_len))
        return ret

    # parses command line arguments
    def parse_args(self, args):
        i = 1
        try:
            inputargs = {}
            self.verbose = 0
            noiputfil = False
            def_path = path.dirname(path.abspath(__file__))
            # TODO: Replace the pre-made panther.in using these defaults.
            altvalues = [  # [idx number, handle, altered value, bool flag(not currently in use), name, default value, comment]
                        [0, "-pfil", "", False, "Pdb file", "prot.pdb", ""],
                        [1, "-rlib", "", False, "Radius library", "{}/rad.lib".format(def_path), ""],
                        [2, "-alib", "", False, "Angle library", "{}/angles.lib".format(def_path), ""],
                        [3, "-chlib", "", False, "Charge library file", "{}/charges.lib".format(def_path), ""],
                        [4, "-cent", "", False, "Center(s)", "0.0 0.0 0.0", "# One or more, separated by ,. May be: coordinates  (format: x y z ), atomIDs (i.e. 1885), resID-atomname or chainID-resnameresID-atomname combinations. If more than one given, the average or the Multibox setting (11) is used."],
                        [5, "-radc", "", False, "Radius center algorithm", "0", "# Create points on all sides of the given center point. 0 = not used. any value above 0.2 = the radius at which the points will be placed"],
                        [6, "-bmp", "", False, "Basic multipoint", "null", "# ChainID-ResID. \"null\" = not used. Center is the average of the given residue. Overrides Centers (5)."],
                        [7, "-frad", "", False, "Filler radius", "0.85", "# Radius of filler atoms. "],
                        [8, "-brad", "", False, "Box radius", "8.0", "# Radius of a box of filler atoms to be created around the center. "],
                        # TODO: Investigate and clarify -bcen
                        [9, "-bcen", "", False, "Box center", "null", "# Forced center point of the filler box. Might not work. \"null\" if not used"],
                        [10, "-mbox", "", False, "Multibox", "Y", "# Y/N. Create a box around each center instead of only one."],
                        [11, "-nem", "", False, "Not empty", "FAD NAP WAT NDP NAI NAD FDA", "# Residue names, separated by a space. (Non-metal-containing) HETATM entries with which pocket atoms should NOT overlap. \"ALL\" means every HETATM and \"null\" will result in all HETATM residues being ignored. "],
                        [12, "-flin", "", False, "Force lining", "null", "# ChainID-ResID, separated by tab or space. Force these amino acids to be included in lining. "],
                        [13, "-ilin", "", False, "Ignore lining", "null", "# ResID, separated by space or tab. Amino acids specified here will be ignored when detecting lining. \"null\" if not used."],
                        [14, "-ofed", "", False, "\"Add\" oxygen at HEM Fe / dual mode", "Y Y", "# Y Y = add oxygen + use dual mode (substrate AND inhibitor models), Y N = add oxygen (substrate model), N N = no charge points for HEM Fe."],
                        [15, "-chrad", "", False, "Charge radius", "0.0", "# Charges for other than optimal pocket atoms will be calculated based on this radius from an otherwise neutral pocket atom to protein atoms"],
                        [16, "-lowch", "", False, "Lowest significant charge (+/-)", "0.16", "# 0.16 = charges between 0.16 and -0.16 will be ignored."],
                        [17, "-watpol", "", False, "Use waters as polar groups", "-1", "# IF >= 0, waters with two hydrogens will be used as polar groups. One day, this might work so that all waters with [number] or more H-bonds to the protein will be used as polar groups and all other waters will be ignored."],
                        [18, "-del", "", False, "Delete farther than", "4.5", "# Delete atoms that are farther than this from protein atoms. 0 if not used."],
                        [19, "-ldlim", "", False, "Ligand distance limit", "null", "# ChainID-ResID Distance OR ChainID Distance. Multiple values allowed, separated by ;. Pocket atoms won't be farther than this from the specified ligand, except for possible charged atoms that are barely outside the limit."],
                        [20, "-fcang", "", False, "False connection angles", "180 90", "# If a group of atoms is connected to the main body with an atom with only two bonds at these angles, the group will be deleted. 180 = only those conencted by a linear bond.  Might work only with cube packing at the moment."],
                        [21, "-fcgrp", "", False, "False connection group size", "10", "# If this is > 0, only the groups from previous setting with MORE THAN this number of atoms will be deleted."],
                        [22, "-ezon", "", False, "Exclusion zone", "null", "# seqID. Multiple allowed (and encouraged), separated by space or tab. The area lined by amino acids listed here will be empty. \"null\" if not used."],
                        [23, "-aex", "", False, "Angle exclusion", "null", ("# Can be used to remove sectors of the pocket. \"null\", \"none\" or \"n\" = not "
                        "used. AA192 15 0 = pocket lining AAs within 15 degrees of center-amino acid with seqid 192 line will be ignored. This is done by "
                        "calculating an average point of all the atoms in the specified amino acid. A122 10 12 5163: All pocket lining amino acids which are "
                        "within 10 degrees of ATOMID5163 - ATOMID122  line will be eliminated IF the distance between them and center is > 12. thus: first "
                        "argument =  AA+seqid or A+atomid, second = degrees. if negative value, angle will be \"directed\" towards the opposite direction, "
                        "third = min distance for deletion, where 0 = unlimited. Last argument can be used to specify the starting point of the angle deletion"
                        " center line. if left blank, the cavity filling center will be used. This is applied before connectness of atoms is checked. Thus if "
                        "deletion of filler atoms leaves two clusters of filler atoms, only the one with the center atom is returned."
                        )],
                        [24, "-pex", "", False, "Plane exclusion", "null", "# Specify 3 atoms to define a plane: atomID OR Chainid-ResnameResid-Atomname, entries separated by \";\". \"null\" if not used. Remove any pocket atoms that are on the different side of the plane than the given/average center point."],
                        [25, "-fpec", "", False, "Force plane exclusion center", "null", "# Coordinates (x y z) OR atomID OR chainid-ResnameResid-Atomname. Force plane exclusion to use this point as a center."],
                        [26, "-gkar", "", False, "Global keep anyway radius", "0", "# Keep atoms within this radius of the pocket volume."],
                        [27, "-kar", "", False, "Keep anyway radius", "4", "# Useful for surface pockets: atoms within this radius will be kept even if they are not within the volume defined by lining amino acids"],
                        [28, "-aalim", "", False, "AA limit", "3", "# If the previous setting is used, the atom still has to be within the aforementioned radius of atoms from at least [number] of amino acids. 0 = not used."],
                        [29, "-slim", "", False, "Specific limits", "null", "# However, they can not be farther than this distance from the specified amino acid (s). AAResid-distance or AAResseq-distance+AAResseq-distance. In the latter case, the two conditions must both apply. Multiple entries allowed, separated by \";\". \"null\" if not used."],
                        # TODO: Investigate whether/how -sec setting works
                        # (and clarify)
                        [30, "-sec", "", False, "Secondary", "N", "# Y/N. Leave in secondary cavities. Might work only with cube packing at the moment."],
                        [31, "-cofil", "", False, "Cofactor fill", "N", "# Y/N. Fill in space occupied by metal cofactors"],
                        [32, "-pack", "", False, "Packing method", "fcc", "# BCC (Body-Centered Cubic, in older versions this was called fcc!), FCC (Face-Centered Cubic) or CUBE (cubic lattice)"],
                        [33, "-creep", "", False, "Creep radius", "null", "# Number. \"null\" if not used. Creeping will start at the lining atom closest to the center. In pocket identification, ignore the atoms that are not within creep radius of the atoms already identified as connected to the central lining."],
                        [34, "-fulli", "", False, "Full lining", "Y", "# Y/N. Get full amino acids when determining lining atoms."],
                        [35, "-adjli", "", False, "Adjacent lining", "N", "# Y/N. Include adjacent amino acids in lining."],
                        [36, "-mbo", "", False, "Multibounds", "Y", "# Base inclusion calculations on all centers, instead of an average point. May help with more complex pockets, but increases calculation time."],
                        [37, "-chdist", "", False, "Max charge-filler distance", "1.6", "# Charged atoms can not be farther than this from the filler pocket atoms."],
                        [38, "-agdist", "", False, "Agonist-distance", "2.5", "# Distance from HEM Fe to antagonist/inhibitor point, and from the latter to agonist/substrate point."],
                        [39, "-atol", "", False, "Angle tolerance", "30", "# Angles for charge atoms can be flexed by this much if the best spot is unavailable."],
                        [40, "-retol", "", False, "Resolution Tolerance", "0.2", "# Wiggleroom to account for resolution etc. Only used when bulding amino acids. "],
                        [41, "-adjdist", "", False, "Adjacent distance", "3", "# Distance for determining which amino acids are adjacent"],
                        [42, "-boinc", "", False, "Boundary increment", "1.3", "# Pocket boundary increment. Smaller values = a LOT more calculations."],
                        [43, "-lidang", "", False, "lining id angle", "35", "# Smaller angle = more accurate lining determination (might be useful with large or surface pockets), but a lot more calculations. Usually a value no smaller than 35 is required."],
                        [44, "-chatrad", "", False, "Radius for charged atoms", "0.2", "# Used for checking overlap with protein atoms."],
                        [45, "-radexdres", "", False, "Min charge-residue distance", "0.4", "# Minimum distance required between charged atom and its residue. Minimum value of 0.5 recommended."],
                        [46, "-hobodist", "", False, "H bond distance", "1.7", "# Atom to hydrogen."],
                        [47, "-donads", "", False, "X-H covalent bond distance", "1.0", "# Length of a covalent bond between hydrogen and its atom"],
                        [48, "-hobomax", "", False, "Max hydrogen bond distance", "4.2", "# Acceptor to donor, not acceptor to hydrogen"]
                        ]
            if "-h" in args:
                self.print_help(altvalues)
                self.quit_now()
            inputargs.update({"output_file": args[-1]})

            inputargs.update({"debug": False})
            inputargs.update({"bounds_only": False})
            inputargs.update({"lining_only": False})
            inputargs.update({"basic_ch_atoms": True})
            #inputargs.update({"no_options": False})
            inputargs.update({"verbose": 0})
            inputargs.update({"centers_only": False})
            inputargs.update({"ch_only": False})
            inputargs.update({"automode": False})
            inputargs.update({"pdbch": False})
            inputargs.update({"signch": False})
            inputargs.update({"cube_only": False})
            inputargs.update({"old_dupl": False})
            inputargs.update({"sectortest": False})

            self.dontname = False


            while i < len(args) - 1:
                if ("-debug" in args[i]):
                    inputargs.update({"debug": True})
                #elif ("-pack" in args[i]):
                #inputargs.update({"pack_method": args[i+1].lower().strip().strip("\"")})
                #    i += 1
                elif ("-noname" in args[i]):
                    self.dontname = True
                elif ("-odupl" in args[i]):
                    inputargs.update({"old_dupl": True})
                elif ("-bo" in args[i]):
                    inputargs.update({"bounds_only": True})
                elif ("--automode" in args[i]):
                    inputargs.update({"automode": True})
                elif ("--pdbch" in args[i]):
                    inputargs.update({"pdbch": True})
                elif ("--signch" in args[i]):
                    inputargs.update({"signch": True})
                elif ("-co" in args[i]):
                    inputargs.update({"centers_only": True})
                elif("-bch" in args[i]):
                    inputargs.update({"basic_ch_atoms": True})
                elif("-ach" in args[i]):
                    inputargs.update({"basic_ch_atoms": False})
                elif ("-defval" in args[i]):
                    noiputfil = True
                elif ("-mkdef" in args[i]):
                    self.print_this("fil", self.get_default_file(altvalues), output=open(args[i+1], 'w'))
                    self.quit_now()
                elif ("-cbo" in args[i]):
                    inputargs.update({"cube_only": True})
                elif ("-sectest" in args[i]):
                    inputargs.update({"sectortest": True})
                elif ("-cho" in args[i]):
                    inputargs.update({"ch_only": True})
                elif ("-lo" in args[i]):
                    inputargs.update({"lining_only": True})
                elif ("-v" in args[i]):
                    inputargs.update({"verbose": 1})
                    self.verbose = inputargs["verbose"]
                else:
                    handled = False
                    for line in altvalues:
                        if args[i].lower() == line[1]:
                            line[2] = args[i+1]
                            i += 1
                            handled = True
                            break
                    if not handled and "iputfile" not in inputargs and not noiputfil:
                        inputargs.update({"iputfile": args[i]})
                i = i + 1
            if ("iputfile" in inputargs):
                inputlines = self.remove_comments(self.read(inputargs["iputfile"]), args=["::", 1])
            else:
                self.print_this("inf", "Loading default values")
                inputlines = self.remove_comments(self.get_default_file(), args=["::", 1])
            for line in altvalues:
                if len(line[2]) > 0:
                    inputlines[line[0]] = line[2]

            inputargs.update({"input_info": inputlines})
            inputargs.update({"valid": True})
            if (len(inputargs["input_info"]) < 19):
                inputargs.update({"valid": False})
            return [inputargs, True]
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            return [inputargs, False, i , args]

    #quit method. Useless now, but exists in case one would prefer to try do
    # something before quitting.
    def quit_now(self):
        sys.exit()

    # if y in line = True
    def true_or_false(self, arg):
        if ("y" in arg.lower()):
            return True
        else:
            return False


    # returns avg points of all amino acids in a given protein chain
    def get_avgs_from_chain(self, chain):
        ret = []
        res = [chain[0]]
        for i in range(1, len(chain)):
            if chain[i].resseq == res[0].resseq:
                res.append(chain[i])
            else:
                ret.append(self.get_avg_point(res))
                res = [chain[i]]
        return ret

    # returns all atoms by residue ID number
    def get_atoms_by_resid(self, idline, atoms, water):
        if (self.is_null(idline)):
            return []
        else:
            idline = int(idline)
        found = False
        ret = []
        for atom in atoms:
            if atom.resseq == idline:
                found = True
                ret.append(atom)
        if not found:
            for atom in water:
                if atom.resseq == idline:
                    ret.append(atom)
        return ret

    # Checks if entry is null. In other words, if value is 0 or "null".
    def is_null(self, entry):
        if type(entry) is int:
            return entry == 0
        if type(entry) is float:
            return entry == 0.0
        if (len(entry) < 1):
            return True
        if type(entry) is str:
            return "null" in entry
        if type(entry) is list:
            return "null" in entry[0]
        return False

    # searches atom list (allatoms) for atoms mentioned in idlines list.
    def fetch_atoms(self, idlines, allatoms, water_included):
        ret = []
        if not self.is_null(idlines):
            searchdict = {}
            id_set = set()
            p = re.compile("\D-")
            water = 0
            resid_atomname_combos = []
            for line in [x.strip() for x in idlines]:
                if (p.match(line) is None):
                    cand = line.split()
                    if (len(cand) > 2):
                        ret.append(Atom("null", 0, "Na", "", "CEN", "", "0", "", float(cand[0]), float(cand[1]), float(cand[2]),"", "0.000", "Na", "0"))
                    else:
                        id_set.add(int(line))
                else:
                    line = line.lower()
                    if(not (len(line) < 2)):
                        data = line.split("-")
                        if (len(data) > 2):
                            searchdict[data[1]] = {}
                            searchdict[data[1]][data[2].strip()] = 1
                            if ("wat" in data[1]):
                                water = water + 1
                        elif(len(data) == 2):
                            resid_atomname_combos.append([data[0], data[1]])
            found = 0
            if(water < len(idlines)):
                for atom in allatoms:
                    if (found == len(idlines)):
                        return ret
                    if (atom.idn in id_set):
                        ret.append(atom)
                        found = found + 1
                    elif(atom.resname.lower() + str(atom.resseq) in searchdict):
                        if(atom.name.strip().lower() in searchdict[atom.resname.lower() + str(atom.resseq)]):
                            if ((data[0] == "no") or (atom.chainid.strip().lower() == data[0])):
                                ret.append(atom)
                                found = found + 1
                    elif(len(resid_atomname_combos) > 0):
                        for entr in resid_atomname_combos:
                            if atom.resseq in entr[0]:
                                if atom.name.strip().lower() in entr[1]:
                                    ret.append(atom)
                                    found = found + 1
            for atom in water_included:
                if (found == len(idlines)):
                    return ret
                if (atom.idn in id_set):
                    ret.append(atom)
                    found = found + 1
                elif (atom.resname.lower() + str(atom.resseq) in searchdict):
                    if(atom.name.strip().lower() in searchdict[atom.resname.lower() + str(atom.resseq)]):
                        if ((data[0] == "no") or (atom.chainid.strip().lower() == data[0])):
                            ret.append(atom)
                            found = found + 1
        return ret

    # parses a charge library file
    def parse_charges(self, libfile, ignore_limit):
        lib = {}
        upper = True
        synonyms = []
        for l in libfile:
            if upper:
                synonyms.append(l.strip("[]").split(","))
            if ("--:--" in l):
                upper = False
                continue
            if not upper:
                line = l.lower().split()
                ch = float(line[-1])
                if (math.fabs(ch) > math.fabs(ignore_limit)):
                    name = line[0].lower().strip()
                    if name not in lib:
                        lib[name] = {}
                    lib[name][line[1].strip("\"").lower().strip()] = ch
        lib["synonyms"] = synonyms
        return lib

    # a method that parses parameters from an input file
    def parse_inputfile(self, inputargs, list_of_lines):
        if ".pdb" not in list_of_lines[0]:
            self.print_this("fat-err", "Wrong file format. \nOnly .pdb accepted for protein structures")
            self.quit_now()

        inputargs.update({"ignore_charge_limit": float(list_of_lines[16])})

        templib = self.remove_comments(self.read(list_of_lines[3]))
        inputargs.update({"charge_lib": self.parse_charges(templib, inputargs["ignore_charge_limit"])})

        delete_chainid = None

        atomlists =  self.parse_pdb(list_of_lines[0], delete_chainid, inputargs["charge_lib"], inputargs["pdbch"], inputargs["signch"])
        inputargs.update({"protatoms": atomlists[0]})
        inputargs.update({"hetatoms": atomlists[1]})
        inputargs.update({"allatoms": atomlists[2]})
        inputargs.update({"chains_dict": atomlists[3]})
        inputargs.update({"hetchains_dict": atomlists[4]})
        inputargs.update({"AA_dict": atomlists[5]})
        inputargs.update({"ligands": atomlists[6]})
        inputargs.update({"AA_original_numbers": atomlists[7]})
        inputargs.update({"water_only": atomlists[8]})
        inputargs.update({"ligands_original_numbers": atomlists[10]})

        if (not inputargs["allatoms"]):
            self.print_this("fat-err", "Empty .pdb file. Check input.")
            self.quit_now()

        if "libfile" not in inputargs:
            inputargs.update({"libfile": list_of_lines[1]})

        try:
            inputargs.update({"distanceinfo": self.remove_comments(self.read(inputargs["libfile"]))})
        except IOError as e:
            self.print_this(e)
            self.print_this("")
            self.print_this("----")
            self.print_this("")
            # Radius file not found
            self.print_this("fat-err",  "Library file " + inputargs["libfile"] + " not found!")
            self.quit_now()

        if "coord_file" not in inputargs:
            inputargs.update({"coord_file": list_of_lines[2]})
        try:
            coor = self.parse_new_coord(self.remove_comments(self.read(inputargs["coord_file"])))
            inputargs.update({"metal_coord": coor[0]})
            inputargs.update({"cofactor_coord": coor[1]})
        except IOError as e:
            self.print_this(e)
            self.print_this("")
            self.print_this("----")
            self.print_this("")
            # Coordination file not found
            self.print_this("fat-err",  "Coordinate file " + inputargs["coord_file"] + " not found!")
            self.quit_now()
        # Radius file does not contain enough lines
        if(len(inputargs["distanceinfo"]) < 1):
            self.print_this("fat-err", "Corrupt radius file.")
            self.quit_now()

        inputargs.update({"outputname": inputargs["output_file"].split(".")[0]})
        # set hascoordinates to True or false and adds centercoordinates.
        inputargs.update({"centers": self.fetch_atoms(list_of_lines[4].strip().lower().split(","), inputargs["allatoms"], inputargs["water_only"])})
        inputargs.update({"fillerradius": float(list_of_lines[7])})
        inputargs.update({"boxradius": float(list_of_lines[8])})

        inputargs.update({"box_center": list_of_lines[9]})
        inputargs.update({"multibox": self.true_or_false(list_of_lines[10])})

        # TODO: Remove twopoint (not used).
        inputargs.update({"twopoint": False})
        if (float(list_of_lines[5])>0.2):
            inputargs["centers"] = self.get_rad_6(inputargs["centers"][0], float(list_of_lines[5]))
        elif (not self.is_null(list_of_lines[6])):
            if list_of_lines[6] in inputargs["ligands_original_numbers"]:
                inputargs["centers"] = [self.get_avg_point(inputargs["ligands_original_numbers"][list_of_lines[6]])]
            elif list_of_lines[6] in inputargs["AA_original_numbers"]:
                inputargs["centers"] = [self.get_avg_point(inputargs["AA_original_numbers"][list_of_lines[6]])]
            else:
                self.print_this("war", "Basic multipoint residue not found: {}\nUsing basic centers: {}".format(list_of_lines[6], list_of_lines[4]))

        if(inputargs["automode"]):
            cent_lig = self.get_bs_non_metal(inputargs["ligands"], inputargs["metal_coord"], list_of_lines[11].lower().split(), True)
            if(len(cent_lig) < 5):
                cent_lig = self.get_bs_metal(inputargs["ligands"], inputargs["metal_coord"], True)
            if(len(cent_lig) < 1):
                cent_lig = self.get_bs_non_metal(inputargs["chains_dict"], inputargs["metal_coord"], list_of_lines[11].split(), False)
                inputargs["centers"] = self.get_avgs_from_chain(cent_lig)
            inputargs["centers"] = self.get_farthest_apart(cent_lig)
            if len(inputargs["centers"]) < 1 or len(inputargs["centers"]) > 25:
                self.print_this("fat-err", "Could not locate proper centers in automode. \nExiting...")
                self.quit_now()
        largest_dist = self.get_largest_closest_distance(inputargs["centers"])
        if (largest_dist / 2) > inputargs["boxradius"]:
            inputargs["boxradius"] = largest_dist / 2 + 0.5
            self.print_this("war", "Box radius less than the largest distance between adjacent center points.\nSetting box radius to {}".format(inputargs["boxradius"]))

        inputargs.update({"donotfillResname": list_of_lines[11].split()})
        inputargs.update({"nofillaage": []})
        if (not self.is_null(inputargs["donotfillResname"][0])):
            inputargs.update({"nofillaage": self.get_resname(inputargs["allatoms"], inputargs["donotfillResname"])})
        inputargs.update({"force_lining":list_of_lines[12].lower().split()})
        inputargs.update({"ignore_lining": list_of_lines[13].split()})

        ago_dual = list_of_lines[14].split()
        inputargs.update({"agon_model": self.true_or_false(ago_dual[0])})
        inputargs.update({"dual_model": self.true_or_false(ago_dual[1])})


        inputargs.update({"charge_radius": float(list_of_lines[15])})
        inputargs.update({"h_bond_limit": int(list_of_lines[17])})
        inputargs.update({"delete_radius": float(list_of_lines[18])})
        inputargs.update({"ligand_distance_restriction": list_of_lines[19]})

        angl = list_of_lines[20].split()
        inputargs.update({"del_angles": []})
        for a in angl:
            inputargs["del_angles"].append(int(a.strip()))
        inputargs.update({"del_min_size": int(list_of_lines[21])})

        inputargs.update({"exclusion_zone": list_of_lines[22].split()})
        angledata = list_of_lines[23].split(";")
        if (not "null" == angledata[0].strip()):
            inputargs.update({"angledata": []})
            for l in angledata:
                inputargs["angledata"].append(l.split())
        else: inputargs.update({"angledata": "null"})


        inputargs.update({"plane_exclusion": list_of_lines[24]})
        if (not self.is_null(inputargs["plane_exclusion"])):
            inputargs.update({"plane_exclusion": self.fetch_atoms(inputargs["plane_exclusion"].split(";"), inputargs["allatoms"], inputargs["water_only"])})
            if (not self.is_null(list_of_lines[25])):
                inputargs.update({"plane_exclusion_center": self.fetch_atoms([list_of_lines[25]], inputargs["allatoms"], inputargs["water_only"])[0]})
            else:
                inputargs.update({"plane_exclusion_center": "null"})

        inputargs.update({"global_keep_anyway_rad": float(list_of_lines[26])})
        inputargs.update({"keep_anyway_radius": float(list_of_lines[27])})
        inputargs.update({"keep_anyway_at_least_AA": int(list_of_lines[28])})
        inputargs.update({"keep_anyway_AA-Resseq-dist": list_of_lines[29]})

        inputargs.update({"keepsecondary": self.true_or_false(list_of_lines[30])})

        inputargs.update({"fill_met_space": self.true_or_false(list_of_lines[31])})

        inputargs.update({"pack_method": list_of_lines[32].lower().strip().strip("\"")})

        temp = list_of_lines[33]
        inputargs.update({"use_creep": True})
        if (not self.is_null(temp)):
            inputargs.update({"creep_radius": float(temp)})
        else: inputargs.update({"use_creep": False})

        inputargs.update({"getFull": self.true_or_false(list_of_lines[34])})
        inputargs.update({"getAdjacent": self.true_or_false(list_of_lines[35])})
        if (len(inputargs["centers"]) < 1):
            if (inputargs["hascoordinates"]):
                inputargs["centers"].append(Atom("null", 1, "O", "", "WAT1", "A", "", "", inputargs["centercoordinates"][0], inputargs["centercoordinates"][1], inputargs["centercoordinates"][2],"", "0.000", "O.3", "0")) # temporary atom for center.
            if (len(inputargs["centers"]) < 1):
                self.print_this("fat-err", "Center atom not found. Check settings")
                self.quit_now()

        inputargs.update({"multibounds": self.true_or_false(list_of_lines[36])})
        inputargs.update({"optimal_inclusion": float(list_of_lines[37])})
        inputargs.update({"agon_dist": float(list_of_lines[38])})

        inputargs.update({"angle_tolerance": float(list_of_lines[39])})
        inputargs.update({"res_tolerance": float(list_of_lines[40])})
        inputargs.update({"AAradius": float(list_of_lines[41])})
        inputargs.update({"pocketIDincrement": float(list_of_lines[42])})

        inputargs.update({"linangle": float(list_of_lines[43])})

        inputargs.update({"wat_conv_distance": None})
        if (list_of_lines[44].lower()[0].isalpha()):
            inputargs.update({"charged_rad": inputargs["fillerradius"]})
        else:
            inputargs.update({"charged_rad": float(list_of_lines[44])})
        inputargs.update({"same_residue_distance_tolerance": float(list_of_lines[45])})
        self.same_residue_distance_tolerance = inputargs["same_residue_distance_tolerance"]
        inputargs.update({"H_bond_distance": float(list_of_lines[46])})
        inputargs.update({"H_add": float(list_of_lines[47])})
        inputargs.update({"hobomax": float(list_of_lines[48])})


    # a method for parsing new coord file format
    def parse_new_coord(self, coord_data):
        cofactor_lib = {}
        metal_lib = {}

        metal_part = False
        for line in coord_data:
            # remove spaces and make the whole thing lowercase.
            line = "".join(line.split()).lower()
            if len(line) > 0:
                if not metal_part:
                    if ("::metals::" in line):
                        metal_part = True
                    else:
                        name, rest_of_line = line.split(":")
                        data_entries = rest_of_line.split(";")
                        actual_data = {}
                        for entr in data_entries:
                            if len(entr) > 0:
                                atom_name, data = entr.strip("]").split("[")
                                actual_data[atom_name] = data.split(",")
                        if name in cofactor_lib:
                            cofactor_lib[name].update(actual_data)
                        else:
                            cofactor_lib[name] = actual_data
                else:
                    atom_element, data = line.strip("]").split("[")
                    metal_lib[atom_element] = data.split(",")
        return [metal_lib, cofactor_lib]

    def onlyClose(self,pocket,lining, limit):
        retlist = []
        for latom in lining:
            add = False;
            for patom in pocket:
                if self.distance(latom, patom) < limit:
                    add = True
                    continue
            if add:
                retlist.append(latom)
        return retlist

    # Main function. Only used for input parsing, launching the actual program
    # and printing out the .mol2 output
    def main(self):
        self.color = { "PURPLE": '\033[95m', "CYAN": '\033[96m', "DARKCYAN": '\033[36m', "BLUE": '\033[94m', "GREEN": '\033[92m', "YELLOW": '\033[93m', "RED": '\033[91m', "BOLD": '\033[1m', "UNDERLINE": '\033[4m', "END": '\033[0m' }
        t = time()
        if len(sys.argv) < 2:
            self.input_required()
        else:
            self.debug_atoms = set()
            parsed = self.parse_args(sys.argv)
            if (not parsed[1]):
                self.print_this("fat-err", "Something went wrong with input! ")
                errstr = "Error token index: " + str(parsed[2]) + "\nInput: " +  " ".join(parsed[3])
                for i in range(0, len(parsed[3])):
                    add = ""
                    end = ""
                    if i == parsed[2]:
                        add = self.color["BOLD"]
                        end = self.color["END"]
                    errstr += "\n" + str(i) + ": " + add + str(parsed[3][i] + end)
                errstr += "\nPerhaps a missing argument / typo / etc?"
                self.print_this("err", errstr)
                self.quit_now()

            iput = parsed[0]
            if(not iput["valid"]):
                self.print_this("err", "Corrupt inputfile.")
                self.quit_now()
            self.debug = iput["debug"]
            if self.debug:
                self.print_this("ver-1", "Debug output enabled")
            self.print_this("ver-1", "Verbose enabled.")
            self.print_this("ver-1", "Parsing input..")
            # add the contents of inputfile to iput
            self.parse_inputfile(iput, iput["input_info"])
            self.distance_dict = {}
            pocket_models = self.findcavity(iput)

            save = self.get_mol2_output(pocket_models, iput["debug"])

            if (iput["iputfile"] == iput["output_file"]):
                self.print_this("war", "Input and output are the same file! \nUsing " + iput["iputfile"].split(".")[0] + "-output.mol2 instead")
                iput["output_file"] = iput["output_file"].split(".")[0] + "-output.mol2"
            elif iput["output_file"].rfind(".") > 0:
                iput["output_file"] = iput["output_file"][:iput["output_file"].rfind(".")]
            self.print_this("inf", "Outputting " + iput["output_file"]+".mol2")
            self.print_this("fil", save, output=open(iput["output_file"]+".mol2", 'w'))

            minu, sec = divmod((time() - t), 60)
            hours, minu = divmod(minu, 60)
            minstr = "minutes"
            secstr = "seconds"
            hstr = "hours"
            if minu < 2:
                minstr = "minute"
            if sec < 2:
                secstr = "second"
            if hours < 2:
                hstr = "hour"

            if hours > 0:
                self.print_this("inf", ("Time elapsed: {0:2} " + hstr + " {0:2} " + minstr + " {0:.2f} " + secstr + ".").format(hours, minu, sec))
            elif minu > 1:
                self.print_this("inf", ("Time elapsed: {0:2} " + minstr + " {0:.2f} " + secstr + ".").format(minu, sec))
            else:
                self.print_this("inf", ("Time elapsed: {0:.2f} " + secstr + ".").format(sec))


if __name__ == "__main__":
    obj = Taskutus()
    obj.main()