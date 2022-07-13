from msmbuilder.utils import load,dump
from msmbuilder.featurizer import DihedralFeaturizer
import os, glob
from msmbuilder.decomposition import tICA
import mdtraj as md
import pandas as pd
import numpy as np

# Load Topology
pdb_file = md.load_pdb('alanine.pdb')
top = pdb_file.topology

# Load Trajectory(s)
trj_list = [md.load_dcd("trajectory.dcd",top=top)]
print("Found %d trajs"%len(trj_list))

# Dihedral Featurizing
f=DihedralFeaturizer(sincos=False)
dump(f,"raw_featurizer.pkl")
feat = f.transform(trj_list)
dump(feat, "raw_features.pkl")

f=DihedralFeaturizer()
dump(f,"featurizer.pkl")
df1 = pd.DataFrame(f.describe_features(trj_list[0]))
print("featurized df shape", df1.shape)
dump(df1,"feature_descriptor.pkl")
feat = f.transform(trj_list)
dump(feat, "features.pkl")

# TICA
t = tICA(lag_time=100,n_components=1,kinetic_mapping=False)
tica_feat = t.fit_transform(feat)
dump(t,"tica_mdl.pkl")
dump(tica_feat,"tica_features.pkl")


class Experiment(object):
    def __init__(self, loc):
        self.loc = loc
        self.descriptors = load("%s/feature_descriptor.pkl" % self.loc)
        self.raw_feat = load("%s/raw_features.pkl" % self.loc)
        self.feat = load("%s//featurizer.pkl" % self.loc)
        self.tica_mdl = load("%s/tica_mdl.pkl" % self.loc)
        self.tica_mdl.weighted_transform = False
        self.tica_mdl.commute_mapping = False
        self.tica_feat = load("%s/tica_features.pkl" % self.loc)


def torsion_label(inds, label):
    #t: TORSION ATOMS=inds
    return "TORSION ATOMS=%s,%s,%s,%s LABEL=%s"%(inds[0],inds[1],inds[2],inds[3],label)+" \\n\\"


def transformer_label(dihedral_feature_label, func, offset):
    #t: TORSION ATOMS=inds
    return "MATHEVAL ARG=%s FUNC=%s(x)-%s LABEL=%s PERIODIC=NO"%(dihedral_feature_label,func,offset,func+"_"+dihedral_feature_label)+" \\n\\"


def mean_free_label(dihedral_feature_label, offset, label):
    return "MATHEVAL ARG=%s FUNC=%s-%s LABEL=%s PERIODIC=NO"\
%(dihedral_feature_label,dihedral_feature_label,offset,"meanfree_"+dihedral_feature_label)+" \\n\\"


def get_indices(dihedral):
    feature_idx = dihedral[0]
    atom_indices = dihedral[1]["atominds"]
    residue_indices = dihedral[1]["resids"]
    otherinfo = dihedral[1]["otherinfo"]
    return feature_idx, atom_indices, residue_indices, otherinfo


def get_dihedral_label_from_residue_indices(dihedral, residue_indices):
    if len(residue_indices) == 2:
        dihedral_label = dihedral[1]["featuregroup"] + "_%s_%s" % (residue_indices[0], residue_indices[1])
    else:
        dihedral_label = dihedral[1]["featuregroup"] + "_%s" % (residue_indices[0])
    return dihedral_label


def write_torsion_labels_to_file(exp: Experiment, indices, file):
    encountered = []
    print("descriptors", exp.descriptors)
    dihedrals = exp.descriptors.iloc[indices].iterrows()
    for dihedral in dihedrals:
        print("dihedral", dihedral)
        feature_idx, atom_indices, residue_indices, otherinfo = get_indices(dihedral)
        dihedral_label = get_dihedral_label_from_residue_indices(dihedral, residue_indices)
        if dihedral_label not in encountered:
            output = torsion_label(atom_indices + 1, label=dihedral_label)  # plumed is 1 indexed and mdtraj is not
            print(output)
            file.writelines(output + "\n")
            encountered.append(dihedral_label)


def write_transform_labels_to_file(exp: Experiment, indices, file):
    encountered = []
    dihedrals = exp.descriptors.iloc[indices].iterrows()
    for dihedral in dihedrals:
        feature_idx, atom_indices, residue_indices, otherinfo = get_indices(dihedral)
        dihedral_label = get_dihedral_label_from_residue_indices(dihedral, residue_indices)
        transform_label = otherinfo + "_" + dihedral_label
        if transform_label not in encountered:
            output = transformer_label(dihedral_label, otherinfo, exp.tica_mdl.means_[feature_idx])
            print(output)
            file.writelines(output + "\n")
            encountered.append(transform_label)


def write_combined_label(exp: Experiment, indices, file):
    combine_args_list = []
    combine_coefficent_list = []
    for j in exp.descriptors.iloc[indices].iterrows():
        feature_index = j[0]
        residue_indices = j[1]["resids"]
        atominds = j[1]["atominds"]
        resids = j[1]["resids"]
        otherinfo = j[1]["otherinfo"]

        if len(resids) == 2:
            dih_label = j[1]["featuregroup"] + "_%s_%s" % (resids[0], resids[1])
        else:
            dih_label = j[1]["featuregroup"] + "_%s" % (resids[0])

        # this is the feature label
        feature_label = otherinfo + "_" + dih_label
        # this is the tic coefficent
        tic_coefficient = exp.tica_mdl.components_[0, feature_index]
        if exp.tica_mdl.kinetic_mapping:
            tic_coefficient *= exp.tica_mdl.eigenvalues_[0]

        # tic_coefficient = a[feature_index]
        label = "tic%d%d" % (0, feature_index)

        combine_args_list.append(feature_label)
        combine_coefficent_list.append(str(tic_coefficient))
    output = "COMBINE LABEL=tic_%d" % 0 + " ARG=%s" % (','.join(combine_args_list)) + \
             " COEFFICIENTS=%s" % (','.join(combine_coefficent_list)) + " PERIODIC=NO" + " \\n\\"
    print(output)
    file.writelines(output + "\n")


def write_final_line(height, pace, file):
    arg_list = []
    sigma_list = []
    arg_list.append("tic_%d" % 0)
    sigma_list.append(str(0.1))

    output = "METAD ARG=%s SIGMA=%s HEIGHT=%s FILE=HILLS PACE=%s LABEL=metad" \
             % (','.join(arg_list), ','.join(sigma_list), \
                str(height), str(pace)) + " \\n\\"
    print(output)
    file.writelines(output + "\n")
    output = "PRINT ARG=%s,metad.bias STRIDE=%s FILE=COLVAR" % (','.join(arg_list), str(pace)) + " \\n"
    print(output)
    file.writelines(output + "\"" + "\n")
    file.close()


def create_plumed_script(exp):
    f = open("./plumed.py", 'w')
    f.write("plumed_script=\"RESTART " + "\\n\\" + "\n")
    print("components", exp.tica_mdl.components_)
    print("eigenvalues", exp.tica_mdl.eigenvalues_)
    inds = np.nonzero(exp.tica_mdl.components_[0, :])
    print("inds", inds)
    write_torsion_labels_to_file(exp, indices=inds, file=f)
    write_transform_labels_to_file(exp, indices=inds, file=f)
    write_combined_label(exp, indices=inds, file=f)
    write_final_line(height=0.2, pace=1000, file=f)


exp = Experiment(loc='./')
create_plumed_script(exp)
