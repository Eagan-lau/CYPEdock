"""CYPEdock single-file filtering/RMSD workflow. No docking or optimization.
Verified protein alignment and atom-mapping functions are preserved below.
Input: the bundled public archive or an explicit --source directory. Output: results.csv and detailed audit files.
"""
from collections import Counter
from types import SimpleNamespace

# ---- Verified geometry and structural IO ----
"""Protein-aligned docking-pose RMSD. No ligand fit, docking, or preparation."""
from pathlib import Path
import hashlib
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

AA = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
              'ARNDCQEGHILKMFPSTWYV'))
NON_REAL_TYPES = {'H','HD','HS','D','G0','G1','G2','G3'}
AD4_ELEMENT = {
    'C':'C','A':'C','CG0':'C','CG1':'C','CG2':'C','CG3':'C',
    'N':'N','NA':'N','NS':'N','O':'O','OA':'O','OS':'O','S':'S','SA':'S',
    'P':'P','F':'F','Cl':'Cl','CL':'Cl','Br':'Br','BR':'Br','I':'I','B':'B',
    'Si':'Si','SI':'Si','Se':'Se','SE':'Se','Zn':'Zn','ZN':'Zn','Fe':'Fe','FE':'Fe',
    'Ca':'Ca','CA':'Ca','Mg':'Mg','MG':'Mg','Mn':'Mn','MN':'Mn','Cu':'Cu','CU':'Cu',
    'Ni':'Ni','NI':'Ni','Co':'Co','CO':'Co','Na':'Na','K':'K'
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_pdb(path, altloc='A'):
    """First model; blank plus explicitly selected alternate, matching the source rule."""
    if altloc is None:altloc='A'
    if not isinstance(altloc,str) or len(altloc)>1:raise ValueError('altloc must be one character or empty')
    result=[]
    for line in Path(path).read_text(encoding='ascii').splitlines():
        if line.startswith('ENDMDL'):break
        if not line.startswith(('ATOM  ','HETATM')):continue
        if len(line)<54:raise ValueError('Truncated coordinate record in '+str(path))
        if line[16] not in (' ',altloc):continue
        symbol=line[76:78].strip() if len(line)>=78 else ''
        if not symbol:
            # Protein names such as CA denote carbon, not calcium.
            name=line[12:16].strip();symbol=next((c for c in name if c.isalpha()),'')
        xyz=np.asarray([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        if not np.isfinite(xyz).all():raise ValueError('Nonfinite PDB coordinates')
        result.append({'record':line[:6].strip(),'name':line[12:16].strip(),
                       'resname':line[17:20].strip(),'chain':line[21],
                       'resseq':int(line[22:26]),'icode':line[26],
                       'altloc':line[16].strip(),'element':symbol,'xyz':xyz,'line':line})
    if not result:raise ValueError('No selected atoms in '+str(path))
    return result

def protein_residues(atoms, chain='A'):
    """Align the full standard-residue sequence first, then select paired C-alpha atoms."""
    groups={}
    for atom in atoms:
        if atom['record']!='ATOM' or atom['chain']!=chain or atom['resname'] not in AA:continue
        key=(atom['resseq'],atom['icode'],atom['resname'])
        group=groups.setdefault(key,{})
        if atom['name'] in group:raise ValueError('Duplicate protein atom after alternate selection: '+str(key))
        group[atom['name']]=atom
    if not groups:raise ValueError('No standard protein residues in chain '+chain)
    return list(groups.items())

def protein_fit(source, reference, excluded_reference_resseq=(), min_pairs=300):
    seq=lambda rows:''.join(AA[key[2]] for key,_ in rows)
    aligner=PairwiseAligner();aligner.mode='global'
    aligner.substitution_matrix=substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score=-10.;aligner.extend_gap_score=-0.5
    alignment=aligner.align(seq(source),seq(reference))[0]
    excluded=set(excluded_reference_resseq);pairs=[]
    for a,b in zip(alignment.aligned[0],alignment.aligned[1]):
        if int(a[1]-a[0])!=int(b[1]-b[0]):raise ValueError('Unequal aligned blocks')
        pairs.extend((int(i),int(j)) for i,j in zip(range(*a),range(*b))
                     if 'CA' in source[i][1] and 'CA' in reference[j][1] and reference[j][0][0] not in excluded)
    if len(pairs)<max(3,min_pairs):raise ValueError('Insufficient paired protein CA atoms: '+str(len(pairs)))
    x=np.asarray([source[i][1]['CA']['xyz'] for i,j in pairs])
    y=np.asarray([reference[j][1]['CA']['xyz'] for i,j in pairs])
    xm,ym=x.mean(0),y.mean(0)
    if np.linalg.matrix_rank(x-xm,tol=1e-10)<2:raise ValueError('Protein CA points are collinear')
    u,_,vt=np.linalg.svd((x-xm).T@(y-ym))
    correction=np.eye(3);correction[-1,-1]=np.sign(np.linalg.det(u@vt))
    rotation=(u@correction@vt).T;translation=ym-xm@rotation.T
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(rotation),1.):
        raise ValueError('Invalid proper rigid transform')
    fitted=x@rotation.T+translation
    ca_rmsd=float(np.sqrt(np.mean(np.sum((fitted-y)**2,axis=1))))
    return rotation,translation,pairs,ca_rmsd

def reference_ligand(atoms,spec,names,elements):
    matches={}
    for atom in atoms:
        if (atom['resname'],atom['chain'],atom['resseq'])!=(spec['resname'],spec['chain'],int(spec['resseq'])):continue
        if atom['element'].upper() in ('H','D'):continue
        if atom['name'] in matches:raise ValueError('Duplicate reference ligand atom: '+atom['name'])
        matches[atom['name']]=atom
    if set(matches)!=set(names):raise ValueError('Reference ligand heavy-atom names do not match metadata')
    if [matches[n]['element'].upper() for n in names]!=[e.upper() for e in elements]:
        raise ValueError('Reference ligand elements do not match metadata')
    return np.asarray([matches[n]['xyz'] for n in names])

def read_models(path):
    """Read and retain exact raw MODEL bytes, including all returned modes."""
    result=[];current=None
    def finish():
        nonlocal current
        if current is None:return
        if not current['atoms'] or 'score' not in current:raise ValueError('MODEL lacks atoms or Vina score')
        current['raw_bytes']=b''.join(current.pop('raw_lines'))
        result.append(current);current=None
    for raw in Path(path).read_bytes().splitlines(keepends=True):
        line=raw.decode('ascii').rstrip('\r\n')
        if line.startswith('MODEL'):
            if current is not None:raise ValueError('Missing ENDMDL before next MODEL')
            current={'mode':int(line.split()[1]),'atoms':{},'raw_lines':[]}
        if current is None:
            if line.startswith(('ATOM  ','HETATM','REMARK VINA RESULT:')):
                current={'mode':1,'atoms':{},'raw_lines':[]}
            else:continue
        current['raw_lines'].append(raw)
        if line.startswith('REMARK VINA RESULT:'):
            current['score']=float(line.split()[3])
            if not np.isfinite(current['score']):raise ValueError('Nonfinite Vina score')
        if line.startswith(('ATOM  ','HETATM')):
            serial=int(line[6:11])
            if serial in current['atoms']:raise ValueError('Duplicate PDBQT serial')
            xyz=np.asarray([float(line[30:38]),float(line[38:46]),float(line[46:54])])
            if not np.isfinite(xyz).all():raise ValueError('Nonfinite PDBQT coordinate')
            current['atoms'][serial]={'serial':serial,'name':line[12:16].strip(),'type':line.split()[-1],'xyz':xyz}
        if line.startswith('ENDMDL'):finish()
    if current is not None:
        if any(line.startswith(b'MODEL') for line in current['raw_lines']):raise ValueError('Unclosed MODEL')
        finish()
    if not result or len({p['mode'] for p in result})!=len(result):raise ValueError('Missing or duplicate modes')
    return result

def map_pose(pose,metadata):
    names=metadata['pdb_atom_names'];qnames=metadata['pdbqt_atom_names'];elements=metadata['elements']
    if not names or len(names)!=len(qnames) or len(names)!=len(elements) or len(set(names))!=len(names) or len(set(qnames))!=len(qnames):
        raise ValueError('Invalid ligand atom-name metadata')
    if 'real_atom_serials' in metadata:
        serials=metadata['real_atom_serials']
        if len(serials)!=len(names) or len(set(serials))!=len(serials):raise ValueError('Invalid real_atom_serials')
        try:atoms=[pose['atoms'][s] for s in serials]
        except KeyError as e:raise ValueError('Mapped real atom missing: '+str(e)) from e
    else:
        byname={}
        for atom in pose['atoms'].values():
            if atom['type'] in NON_REAL_TYPES:continue
            if atom['name'] in byname:raise ValueError('Ambiguous repeated PDBQT atom name')
            byname[atom['name']]=atom
        if set(byname)!=set(qnames):raise ValueError('Unmapped or missing PDBQT real heavy atoms')
        atoms=[byname[n] for n in qnames]
    if [a['name'] for a in atoms]!=qnames:raise ValueError('PDBQT serial-to-name mapping differs')
    if any(a['type'] in NON_REAL_TYPES for a in atoms):raise ValueError('Hydrogen or glue pseudo-atom in real mapping')
    for atom,element in zip(atoms,elements):
        if atom['type'] not in AD4_ELEMENT:raise ValueError('Unknown PDBQT atom type: '+atom['type'])
        if AD4_ELEMENT[atom['type']].upper()!=element.upper():raise ValueError('PDBQT atom type differs from mapped element')
    # CG0 is a real carbon type; only G0-G3 pseudo-atoms are excluded.
    real={s for s,a in pose['atoms'].items() if a['type'] not in NON_REAL_TYPES}
    if real!={a['serial'] for a in atoms}:raise ValueError('Real heavy-atom serial mapping is incomplete')
    return np.asarray([a['xyz'] for a in atoms]),atoms

def validate_permutations(metadata):
    elements=metadata['elements'];n=len(elements)
    values=metadata.get('symmetry_permutations')
    if not values:raise ValueError('Explicit permitted symmetry permutations are required')
    result=[]
    for perm in values:
        if not isinstance(perm,list) or any(type(i) is not int for i in perm) or sorted(perm)!=list(range(n)):
            raise ValueError('Invalid symmetry permutation')
        if [elements[i].upper() for i in perm]!=[e.upper() for e in elements]:raise ValueError('Symmetry permutation swaps elements')
        result.append(np.asarray(perm,dtype=int))
    return result

def ligand_rmsd(reference,xyz,permutations):
    if reference.shape!=xyz.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid ligand coordinate arrays')
    fixed=float(np.sqrt(np.mean(np.sum((reference-xyz)**2,axis=1))))
    values=[float(np.sqrt(np.mean(np.sum((reference-xyz[p])**2,axis=1)))) for p in permutations]
    index=int(np.argmin(values));return values[index],fixed,index

def receptor_fe(atoms):
    fe=[a['xyz'] for a in atoms if a['resname'] in ('HEM','HEC') and a['name']=='FE']
    if len(fe)!=1:raise ValueError('Exactly one HEM/HEC FE is required for the declared hit annotation')
    return fe[0]

def ligand_pdb_text(xyz,names,elements,resname,first_serial=1):
    return ''.join(f'HETATM{first_serial+i:5d} {n:>4s} {resname:3s} Z9999    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}\n'
                   for i,(n,e,(x,y,z)) in enumerate(zip(names,elements,xyz)))

def receptor_pdb_text(atoms,rotation=None,translation=None):
    result=[]
    for atom in atoms:
        line=atom['line']
        if rotation is not None:
            xyz=atom['xyz']@rotation.T+translation
            line=line[:30]+''.join(f'{v:8.3f}' for v in xyz)+line[54:]
        result.append(line+'\n')
    return ''.join(result)

def export_selected(folder,pose,raw_xyz,metadata,spec,receptor_atoms,rotation,translation,input_sdf):
    """Export selected coordinates without chemical repair or minimization."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    names,elements=metadata['pdb_atom_names'],metadata['elements']
    moved=raw_xyz@rotation.T+translation
    receptor=receptor_pdb_text(receptor_atoms)
    start=max(int(a['line'][6:11]) for a in receptor_atoms)+1
    files={'receptor.pdb':receptor+'END\n',
           'ligand.pdb':ligand_pdb_text(raw_xyz,names,elements,spec['resname'])+'END\n',
           'complex.pdb':receptor+'TER\n'+ligand_pdb_text(raw_xyz,names,elements,spec['resname'],start)+'END\n',
           'ligand_crystal_frame.pdb':ligand_pdb_text(moved,names,elements,spec['resname'])+'END\n',
           'complex_crystal_frame.pdb':receptor_pdb_text(receptor_atoms,rotation,translation)+'TER\n'+ligand_pdb_text(moved,names,elements,spec['resname'],start)+'END\n'}
    for name,text in files.items():
        with (folder/name).open('x',encoding='ascii',newline='\n') as f:f.write(text)
    with (folder/'selected_raw_model.pdbqt').open('xb') as f:f.write(pose['raw_bytes'])
    source=Chem.SDMolSupplier(str(input_sdf),removeHs=False)[0]
    if source is None:raise ValueError('Input chemical SDF cannot be read')
    mol=Chem.RemoveHs(Chem.Mol(source))
    if [a.GetSymbol() for a in mol.GetAtoms()]!=elements:raise ValueError('Input SDF heavy-atom order differs from metadata')
    if 'formal_charge' in metadata and Chem.GetFormalCharge(mol)!=metadata['formal_charge']:raise ValueError('SDF formal charge differs')
    mol.RemoveAllConformers();conf=Chem.Conformer(len(names));conf.Set3D(True)
    for i,coord in enumerate(raw_xyz):conf.SetAtomPosition(i,Point3D(*map(float,coord)))
    mol.AddConformer(conf,assignId=True);mol.SetProp('_Name',folder.name)
    mol.SetProp('ATOM_NAMES',','.join(names));mol.SetProp('COORDINATE_POLICY','Selected raw pose; no ligand-only fit or optimization')
    writer=Chem.SDWriter(str(folder/'ligand.sdf'));writer.SetKekulize(True);writer.write(mol);writer.close()
    check=Chem.SDMolSupplier(str(folder/'ligand.sdf'),removeHs=False)[0]
    if check is None:raise ValueError('Selected SDF readback failed')
    if not np.allclose(check.GetConformer().GetPositions(),raw_xyz,rtol=0,atol=0.000051):raise ValueError('SDF coordinates changed on export')
    return {'raw_ligand_xyz':raw_xyz.tolist(),'crystal_frame_ligand_xyz':moved.tolist(),
            'rotation':rotation.tolist(),'translation':translation.tolist(),'no_ligand_fit':True,
            'source_input_isomeric_smiles':Chem.MolToSmiles(Chem.RemoveHs(source),isomericSmiles=True),
            'export_readback_isomeric_smiles':Chem.MolToSmiles(check,isomericSmiles=True),
            'chemical_QA_status':'NOT_ASSESSED_BY_FORMAT_EXPORT'}

# Capture the unchanged numerical helper functions in their original namespace.
core = SimpleNamespace(**{name: globals()[name] for name in ['sha256', 'read_pdb', 'protein_residues', 'protein_fit', 'reference_ligand', 'read_models', 'map_pose', 'validate_permutations', 'ligand_rmsd', 'receptor_fe', 'ligand_pdb_text', 'receptor_pdb_text', 'export_selected']})

# ---- Verified all-mode evaluation ----
"""Evaluate all supplied docking modes and keep score and RMSD selection separate."""
import argparse,csv,json,re,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def readj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def csvout(p,rows):
    if not rows:return
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def safe_label(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value) or value in ('.','..'):
        raise ValueError('Unsafe output label: '+str(value))
    return value
def scorekey(row):return (row['vina_kcal_mol'],row['docking_seed'],row['mode'],row['record_id'])
def oraclekey(row):return (row['rmsd_symmetry_A'],*scorekey(row))

def evaluate(manifest_path,output,min_ca_pairs=300,fe_cutoff=5.,score_cutoff=-7.,verify_tolerance=1e-8,state_manifest_path=None):
    from rdkit import Chem  # Check export dependency before creating any output.
    manifest_path=Path(manifest_path).resolve();manifest=readj(manifest_path);base=manifest_path.parent
    records=manifest.get('records')
    if not isinstance(records,list) or not records:raise ValueError('Manifest records must be a nonempty list')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Output already exists; choose a new directory: '+str(output))
    input_bindings={str(manifest_path):core.sha256(manifest_path)}
    def bind(name):
        p=Path(name);p=p if p.is_absolute() else base/p;p=p.resolve()
        if not p.is_file():raise FileNotFoundError(str(p))
        digest=core.sha256(p)
        if str(p) in input_bindings and input_bindings[str(p)]!=digest:raise ValueError('Input changed while reading')
        input_bindings[str(p)]=digest;return p
    # Validate package-level input bindings before computing anything.
    provided=base/'input_sha256.json'
    if provided.exists():
        content=readj(provided)
        entries=content.get('files',content)
        if isinstance(entries,dict):
            for name,info in entries.items():
                expected=info['sha256'] if isinstance(info,dict) else info
                p=bind(name)
                if core.sha256(p)!=expected:raise ValueError('Bundled input SHA mismatch: '+name)
        input_bindings[str(provided)]=core.sha256(provided)
    states_path=Path(state_manifest_path).resolve() if state_manifest_path else base/manifest.get('receptor_states_manifest','receptor_states.json')
    state_registry={}
    if states_path.is_file():
        states_path=bind(str(states_path));state_document=readj(states_path)
        for state in state_document.get('states',[]):
            if 'state_json' in state:
                source_path=Path(state['state_json'])
                if not source_path.is_absolute():source_path=states_path.parent/source_path
                source_path=bind(str(source_path));source_state=readj(source_path)
                if state.get('state_json_sha256') and core.sha256(source_path)!=state['state_json_sha256']:
                    raise ValueError('Source receptor-state SHA mismatch')
                if any(key in state for key in ('case','state_id','status','gate')):
                    raise ValueError('A state_json wrapper must not override source identity, status, or gate')
                extras={k:v for k,v in state.items() if k not in ('state_json','state_json_sha256')}
                if 'cohort' in extras and 'cohort' in source_state and extras['cohort']!=source_state['cohort']:
                    raise ValueError('Source and wrapper cohorts disagree')
                state={**source_state,**extras}
            key=(state.get('cohort',''),state['case'],state['state_id'])
            if key in state_registry:raise ValueError('Duplicate source receptor-state key')
            state_registry[key]=state
    output.mkdir(parents=True)
    allrows=[];contexts={};checks=[];failures=[];record_ids=set();alignments=[]
    physical_cache={}
    for index,record in enumerate(records,1):
        cid=safe_label(record['cell_id']);rid=safe_label(record.get('record_id',cid+'__'+str(index)))
        if rid in record_ids:raise ValueError('Duplicate record_id: '+rid)
        record_ids.add(rid)
        try:
            seed=record.get('docking_seed')
            if type(seed) is not int:raise ValueError('Explicit integer docking_seed is required, independent of expected checks')
            paths={k:bind(record[k]) for k in ['receptor_pdb','reference_pdb','ligand_metadata_json','raw_poses_pdbqt','ligand_input_sdf']}
            meta=readj(paths['ligand_metadata_json']);spec=record['reference']
            ref_atoms=core.read_pdb(paths['reference_pdb'],spec.get('altloc','A'))
            rec_atoms=core.read_pdb(paths['receptor_pdb'],record.get('receptor_altloc','A'))
            source=core.protein_residues(rec_atoms,record.get('protein_chain','A'))
            target=core.protein_residues(ref_atoms,record.get('reference_protein_chain','A'))
            rotation,translation,pairs,ca_rmsd=core.protein_fit(source,target,spec.get('excluded_crystal_resseq',[]),min_ca_pairs)
            names,elements=meta['pdb_atom_names'],meta['elements']
            reference=core.reference_ligand(ref_atoms,spec,names,elements)
            permutations=core.validate_permutations(meta)
            secondary=None
            if record.get('case')=='3AWM' or spec.get('secondary_altloc'):
                secondary=core.reference_ligand(core.read_pdb(paths['reference_pdb'],spec.get('secondary_altloc','B')),spec,names,elements)
            fe=core.receptor_fe(rec_atoms)
            reaction=meta['reaction_index']
            if type(reaction) is not int or not 0<=reaction<len(names):raise ValueError('Invalid reaction atom index')
            gate=record.get('gate_passed');ready=record.get('receptor_status')=='READY';eligible=ready and gate is True
            arm=record.get('arm','UNSPECIFIED')
            state_key=(record.get('cohort',''),record.get('case',''),record['state_id'])
            actual=state_registry.get(state_key)
            if state_registry and actual is None:raise ValueError('Record has no matching original source receptor state')
            if actual is not None and (actual.get('status')!=record.get('receptor_status') or
                                      actual.get('gate',{}).get('passed') is not gate):
                raise ValueError('Record gate/status differs from original receptor-state evidence')
            if arm=='E' and actual is None:raise ValueError('E records require original receptor-state manifest evidence')
            if not ready:raise ValueError('Receptor is not READY; retained as failure, not silently dropped')
            if arm=='E' and not eligible:raise ValueError('E record violates the declared READY + strict gate=True subset')
            digest=core.sha256(paths['raw_poses_pdbqt'])
            if digest not in physical_cache:physical_cache[digest]=core.read_models(paths['raw_poses_pdbqt'])
            poses=physical_cache[digest]
            context={'record':record,'metadata':meta,'spec':spec,'reference':reference,'rotation':rotation,'translation':translation,
                     'receptor_atoms':rec_atoms,'paths':paths,'poses':{p['mode']:p for p in poses}}
            alignment_row={'record_id':rid,'cell_id':cid,'state_id':record['state_id'],'matched_CA':len(pairs),
                'protein_CA_RMSD_A':ca_rmsd,'rotation':rotation.tolist(),'translation':translation.tolist(),
                'transform_convention':'reference_xyz = receptor_xyz @ rotation.T + translation',
                'excluded_reference_resseq':spec.get('excluded_crystal_resseq',[]),
                'source_resseq':[source[i][0][0] for i,j in pairs],'reference_resseq':[target[j][0][0] for i,j in pairs],
                'ligand_only_fit':False}
            record_rows=[]
            for pose in poses:
                xyz,_=core.map_pose(pose,meta);moved=xyz@rotation.T+translation
                sym,fixed,pi=core.ligand_rmsd(reference,moved,permutations)
                secondary_value=core.ligand_rmsd(secondary,moved,permutations)[0] if secondary is not None else None
                distance=float(np.linalg.norm(xyz[reaction]-fe))
                row={'record_id':rid,'cell_id':cid,'case':record.get('case',''),'arm':arm,'physical_source_arm':record.get('physical_source_arm',arm),
                     'cohort':record.get('cohort','UNSPECIFIED'),'state_id':record['state_id'],'docking_seed':seed,'mode':pose['mode'],
                     'vina_kcal_mol':pose['score'],'reaction_Fe_A':distance,'Fe_score_hit':distance<fe_cutoff and pose['score']<score_cutoff,
                     'receptor_status':record.get('receptor_status'),'gate_passed':gate,'E_eligible_receptor':eligible,'E_alias':arm=='E',
                     'rmsd_symmetry_A':sym,'rmsd_fixed_name_A':fixed,'secondary_reference_symmetry_A':secondary_value,
                     'symmetry_permutation_index':pi,'heavy_atom_n':len(names),'matched_protein_CA':len(pairs),'protein_CA_RMSD_A':ca_rmsd,
                     'physical_call_sha256':digest,'physical_pose_uid':digest+':mode_'+str(pose['mode']),
                     'chemical_QA_status':'NOT_ASSESSED_BY_THIS_RMSD_WORKFLOW'}
                record_rows.append(row)
            # A malformed later MODEL invalidates this whole supplied call.
            # Commit neither partial rows nor a successful context until all modes pass.
            contexts[rid]=context;alignments.append(alignment_row);allrows.extend(record_rows)
            expected=record.get('selected_expected')
            if expected:
                found=next((r for r in allrows if r['record_id']==rid and r['mode']==expected['mode']),None)
                passed=found is not None and expected['seed']==seed
                sym_delta=abs(found['rmsd_symmetry_A']-expected['RMSD_symmetry_A']) if found else None
                fixed_delta=abs(found['rmsd_fixed_name_A']-expected['fixed_name_A']) if found else None
                passed=passed and sym_delta<=verify_tolerance and fixed_delta<=verify_tolerance
                checks.append({'record_id':rid,'cell_id':cid,'expected_mode':expected['mode'],'seed_matches':expected['seed']==seed,
                               'symmetry_absolute_delta_A':sym_delta,'fixed_name_absolute_delta_A':fixed_delta,'passed':bool(passed)})
        except Exception as e:
            failures.append({'record_id':rid,'cell_id':cid,'state_id':record.get('state_id'),'error':type(e).__name__+': '+str(e)})
    summaries=[];selections=[]
    for cid in sorted({r['cell_id'] for r in allrows}):
        pool=[r for r in allrows if r['cell_id']==cid];hits=[r for r in pool if r['Fe_score_hit']]
        groups={(r['case'],r['arm'],r['cohort']) for r in pool}
        if len(groups)!=1:
            failures.append({'cell_id':cid,'error':'A cell_id mixes cases, arms or cohorts; no group selection exported'})
            continue
        winners={'all_pose_rmsd_min':min(pool,key=oraclekey),'score_top1':min(pool,key=scorekey),
                 'Fe_score_rmsd_min':min(hits,key=oraclekey) if hits else None,
                 'Fe_score_score_best':min(hits,key=scorekey) if hits else None}
        summary={'cell_id':cid,'evaluated_record_n':len({r['record_id'] for r in pool}),'evaluated_state_n':len({r['state_id'] for r in pool}),
                 'evaluated_arm_pose_n':len(pool),'evaluated_unique_physical_pose_n':len({r['physical_pose_uid'] for r in pool}),
                 'Fe_score_hit_n_in_supplied_pool':len(hits),'minimum_scope':manifest.get('population_scope','USER_SUPPLIED_RECORDS'),
                 'all_pose_rmsd_min_A':winners['all_pose_rmsd_min']['rmsd_symmetry_A'],
                 'fixed_name_RMSD_of_symmetry_winner_A':winners['all_pose_rmsd_min']['rmsd_fixed_name_A'],
                 'independent_fixed_name_min_A':min(r['rmsd_fixed_name_A'] for r in pool),
                 'score_top1_RMSD_A':winners['score_top1']['rmsd_symmetry_A'],
                 'Fe_score_rmsd_min_A':winners['Fe_score_rmsd_min']['rmsd_symmetry_A'] if hits else None,
                 'raw_Vina_top5_best_RMSD_A':min(r['rmsd_symmetry_A'] for r in sorted(pool,key=scorekey)[:5]),
                 'scenario_hit_rate':'NOT_ESTIMATED_UNLESS_COMPLETE_POPULATION_PROVIDED',
                 'source_declared_scenario_pose_n':contexts[pool[0]['record_id']]['record'].get('source_scenario_pose_count'),
                 'source_declared_scenario_call_n':contexts[pool[0]['record_id']]['record'].get('source_scenario_call_count')}
        summaries.append(summary)
        for label,winner in winners.items():
            if winner is None:continue
            context=contexts[winner['record_id']];pose=context['poses'][winner['mode']]
            xyz,_=core.map_pose(pose,context['metadata'])
            try:
                export=core.export_selected(output/'selected'/cid/label,pose,xyz,context['metadata'],context['spec'],context['receptor_atoms'],
                                            context['rotation'],context['translation'],context['paths']['ligand_input_sdf'])
                dump(output/'selected'/cid/label/'selection.json',{'criterion':label,'retrospective_native_reference_used':label.endswith('rmsd_min'),
                     'pose':winner,'coordinates_and_export':export,'note':'Chemical validity was not re-assessed by this RMSD-only export.'})
                selections.append({'cell_id':cid,'selection':label,**winner})
            except Exception as e:
                failures.append({'cell_id':cid,'selection':label,'phase':'EXPORT_FAILED_PARTIAL_FILES_PRESERVED',
                                 'error':type(e).__name__+': '+str(e)})
    csvout(output/'per_pose.csv',allrows);csvout(output/'summary.csv',summaries)
    csvout(output/'selected_poses.csv',selections);csvout(output/'expected_checks.csv',checks)
    dump(output/'summary.json',summaries);dump(output/'alignments.json',alignments);dump(output/'failures.json',failures)
    dump(output/'input_sha256.json',input_bindings)
    changed=[p for p,digest in input_bindings.items() if not Path(p).is_file() or core.sha256(p)!=digest]
    failed_checks=[r for r in checks if not r['passed']]
    status='PASS_COMPLETE_SUPPLIED_RECORD_EVALUATION' if not (failures or changed or failed_checks) else 'FAILED_OUTPUTS_PRESERVED'
    receipt={'status':status,'created_utc':datetime.now(timezone.utc).isoformat(),'input_manifest':str(manifest_path),
             'population_scope':manifest.get('population_scope','USER_SUPPLIED_RECORDS'),'supplied_record_n':len(records),
             'successful_record_n':len(contexts),'failure_n':len(failures),'evaluated_arm_pose_n':len(allrows),
             'unique_physical_call_n':len({r['physical_call_sha256'] for r in allrows}),
             'unique_physical_pose_n':len({r['physical_pose_uid'] for r in allrows}),
             'expected_check_n':len(checks),'failed_expected_check_n':len(failed_checks),'source_files_changed':changed,
             'min_CA_pairs_required':min_ca_pairs,'Fe_score_hit_rule':{'reaction_Fe_A_strict_less_than':fe_cutoff,'vina_kcal_mol_strict_less_than':score_cutoff},
             'RMSD_definition':'Whole-protein CA-aligned heavy atoms; fixed-name and allowed-symmetry endpoints separate; no ligand-only fitting',
             'primary_reference':'Declared altloc; secondary altloc never selected as the minimum primary reference',
             'original_E_subset_enforced':True,'outputs_are_postprocessing_only':True,'docking_or_optimization_performed':False,
             'chemical_QA':'NOT_ASSESSED; original graph is used for export, not a chemical-validity certificate'}
    dump(output/'completion_receipt.json',receipt)
    return receipt

# ---- Verified stored-state filter ----
RULE = "status == 'READY' and gate.passed is True"

def decide(state, source_path, source_digest, manifest_position=None):
    if not isinstance(state, dict):
        raise ValueError("An original state receipt must be a JSON object")
    sid = state.get("state_id")
    case = state.get("case", state.get("case_id"))
    if not isinstance(sid, str) or not sid or not isinstance(case, str) or not case:
        raise ValueError("Every state requires nonempty case (or case_id) and state_id strings")
    status = state.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("Missing original preparation status for " + sid)
    gate = state.get("gate")
    present = isinstance(gate, dict) and "passed" in gate
    passed = gate.get("passed") if isinstance(gate, dict) else None
    ready = status == "READY"
    retained = ready and passed is True
    if not ready:
        reason = "EXCLUDED_PREPARATION_STATUS_" + status
    elif not present:
        reason = "EXCLUDED_GATE_NOT_RECORDED"
    elif passed is True:
        reason = "RETAINED_READY_AND_STRICT_GATE_TRUE"
    elif passed is False:
        reason = "EXCLUDED_GATE_FALSE"
    elif passed is None:
        reason = "EXCLUDED_GATE_NULL_NOT_ASSESSED"
    else:
        reason = "EXCLUDED_GATE_NOT_A_JSON_BOOLEAN"
    row = dict(case=case, state_id=sid, status=status,
               gate_passed_json=json.dumps(passed, ensure_ascii=True, allow_nan=False),
               gate_value_type=type(passed).__name__ if present else "MISSING",
               D_ready=ready, E_retained=retained, decision_reason=reason,
               source_state_json=source_path, source_sha256=source_digest,
               protein=state.get("protein", ""), receptor=state.get("receptor", ""),
               pdbqt=state.get("pdbqt", ""))
    if manifest_position is not None:
        row["source_manifest_index"] = manifest_position
    return row

def summarize(rows, seeds):
    all_n = len(rows)
    ready_n = sum(row["D_ready"] for row in rows)
    retained_n = sum(row["E_retained"] for row in rows)
    result = dict(input_sampled_state_n=all_n, D_ready_state_n=ready_n,
                  non_READY_state_n=all_n-ready_n, E_retained_state_n=retained_n,
                  D_ready_but_gate_excluded_state_n=ready_n-retained_n,
                  gate_true_but_non_READY_state_n=sum(row["gate_passed_json"] == "true" and not row["D_ready"] for row in rows),
                  E_retained_fraction_of_sampled=retained_n/all_n if all_n else None,
                  E_retained_fraction_of_D_ready=retained_n/ready_n if ready_n else None,
                  status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
                  decision_counts=dict(sorted(Counter(row["decision_reason"] for row in rows).items())))
    result["declared_docking_seeds"] = seeds
    result["declared_seeds_per_state"] = len(seeds) if seeds is not None else None
    result["planned_call_slots_sampled_mother_pool"] = all_n*len(seeds) if seeds is not None else None
    result["D_ready_call_slots"] = ready_n*len(seeds) if seeds is not None else None
    result["E_retained_call_slots"] = retained_n*len(seeds) if seeds is not None else None
    result["call_accounting_note"] = "Declared slots only; not completed searches or returned poses. Unknown if seeds were not supplied."
    return result

# ---- Public entry point: validate and extract the bundled selected-call archive. ----
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

def _extract_input_archive(archive, destination):
    """Extract only hash-bound regular members; reject traversal, links and unexpected entries."""
    import stat
    import zipfile
    from pathlib import PurePosixPath
    with zipfile.ZipFile(archive) as bundle:
        entries=bundle.infolist()
        names=[entry.filename for entry in entries]
        if len(names)!=len(set(names)) or len(names)>1000:
            raise ValueError('Duplicate or excessive archive members')
        if sum(entry.file_size for entry in entries)>200*1024*1024:
            raise ValueError('Archive is larger than the supported selected-call dataset')
        for entry in entries:
            name=entry.filename;parts=PurePosixPath(name).parts
            if (not parts or name.startswith('/') or ':' in name or chr(92) in name or
                    any(part in ('..','.') for part in parts) or entry.is_dir() or
                    stat.S_ISLNK(entry.external_attr>>16) or entry.file_size>32*1024*1024):
                raise ValueError('Unsafe archive member: '+name)
        manifest=json.loads(bundle.read('archive_manifest.json'))
        members=manifest.get('members',{})
        if manifest.get('schema')!=1 or manifest.get('cell_n')!=32 or len(members)!=manifest.get('member_n'):
            raise ValueError('Unsupported archive manifest')
        if set(names)!=set(members)|{'archive_manifest.json'}:
            raise ValueError('Archive entries differ from the bound member manifest')
        destination.mkdir(parents=True,exist_ok=False)
        for entry in entries:
            value=bundle.read(entry)
            if entry.filename!='archive_manifest.json':
                binding=members[entry.filename]
                if len(value)!=binding['bytes'] or hashlib.sha256(value).hexdigest()!=binding['sha256']:
                    raise ValueError('Archive member digest mismatch: '+entry.filename)
            target=destination.joinpath(*PurePosixPath(entry.filename).parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as handle:handle.write(value)
    return destination


def _source_records(source):
    records=[];states={};source_hashes={}
    for metadata_path in sorted(source.glob('*/cell_metadata.json')):
        folder=metadata_path.parent;metadata=readj(metadata_path)
        cid=safe_label(metadata['cell_id'])
        if cid!=folder.name:raise ValueError('Cell ID and source folder disagree')
        for name,expected in metadata['file_sha256'].items():
            path=folder/name
            if Path(name).name!=name or not path.is_file() or core.sha256(path)!=expected:
                raise ValueError('Original source file missing or changed: '+str(path))
            source_hashes[str(path.resolve())]=expected
        source_hashes[str(metadata_path.resolve())]=core.sha256(metadata_path)
        state=readj(folder/'original_state.json')
        ligand=readj(folder/'original_ligand_metadata.json')
        alignment=readj(folder/'atom_mapping_and_alignment.json')
        argv=readj(folder/'original_command_argv.json')
        seed=int(argv[argv.index('--seed')+1])
        if seed!=metadata['actual_vina_seed'] or argv!=metadata['original_command_argv']:
            raise ValueError('Original docking command/seed differs from Cell metadata')
        if state['case']!=metadata['case'] or state['state_id']!=metadata['state_id']:
            raise ValueError('Original receptor-state identity differs from Cell metadata')
        if metadata['arm'] in ('E','F') and not (state['status']=='READY' and state.get('gate',{}).get('passed') is True):
            raise ValueError('Retained-state E/F record lacks a READY, gate-passed receptor')
        # Bind the original native ligand by its preserved coordinates, not by
        # an arbitrary first residue, proximity, or best prediction RMSD.
        names=ligand['pdb_atom_names'];elements=ligand['elements']
        native_copy=core.read_pdb(folder/'ligand_reference_only.pdb')
        copy_by_name={a['name']:a for a in native_copy if a['element'].upper() not in ('H','D')}
        if set(copy_by_name)!=set(names):raise ValueError('Incomplete preserved native-ligand copy')
        native_xyz=np.asarray([copy_by_name[n]['xyz'] for n in names])
        if not np.array_equal(native_xyz,np.asarray(alignment['reference_xyz'])):
            raise ValueError('Preserved native coordinates differ from frozen atom mapping')
        altloc=alignment.get('primary_altloc','A')
        crystal=core.read_pdb(folder/'crystal_reference.pdb',altloc)
        groups={}
        for atom in crystal:
            if atom['resname']!=metadata['ligand_CCD'] or atom['element'].upper() in ('H','D'):continue
            key=(atom['resname'],atom['chain'],atom['resseq'],atom['icode'])
            group=groups.setdefault(key,{})
            if atom['name'] in group:raise ValueError('Ambiguous reference-ligand alternate atoms')
            group[atom['name']]=atom
        matches=[]
        for key,atoms in groups.items():
            if set(atoms)!=set(names):continue
            if [atoms[n]['element'].upper() for n in names]!=[e.upper() for e in elements]:continue
            xyz=np.asarray([atoms[n]['xyz'] for n in names])
            if np.array_equal(xyz,native_xyz):matches.append(key)
        if len(matches)!=1:raise ValueError('Native reference cannot be uniquely bound to its preserved coordinates')
        resname,chain,resseq,icode=matches[0]
        if icode.strip():raise ValueError('Reference insertion code requires an explicit supported mapping')
        spec={'resname':resname,'chain':chain,'resseq':resseq,'altloc':altloc,
              'excluded_crystal_resseq':alignment.get('alignment_excluded_crystal_resseq',[])}
        if metadata['case']=='3AWM':spec['secondary_altloc']='B'
        record={'record_id':cid+'__'+metadata['pose_id'].split('/mode_')[0],
                'cell_id':cid,'case':metadata['case'],'arm':metadata['arm'],
                'physical_source_arm':metadata['physical_source_arm'],'cohort':metadata['source_cohort'],
                'state_id':metadata['state_id'],'docking_seed':seed,
                'receptor_status':state['status'],'gate_passed':state.get('gate',{}).get('passed'),
                'receptor_pdb':str((folder/'receptor_docking_frame.pdb').resolve()),
                'reference_pdb':str((folder/'crystal_reference.pdb').resolve()),
                'ligand_input_sdf':str((folder/'ligand_input.sdf').resolve()),
                'ligand_metadata_json':str((folder/'original_ligand_metadata.json').resolve()),
                'raw_poses_pdbqt':str((folder/'all_raw_models.pdbqt').resolve()),'reference':spec,
                'selected_expected':{'mode':metadata['selected_mode'],'seed':seed,
                                     'RMSD_symmetry_A':metadata['exact_symmetry_RMSD_A'],
                                     'fixed_name_A':metadata['fixed_name_RMSD_A']},
                'source_scenario_pose_count':metadata['scenario_total_poses'],
                'source_scenario_call_count':metadata['scenario_total_calls']}
        records.append(record)
        state=dict(state,cohort=metadata['source_cohort'])
        key=(state['cohort'],state['case'],state['state_id'])
        if key in states and states[key]!=state:raise ValueError('Conflicting original state records')
        states[key]=state
    if not records:raise ValueError('No complete Cell archives found under --source')
    return records,list(states.values()),source_hashes

def _filter_original_states(states,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=[dict(decide(s,'original_state.json','',i),cohort=s.get('cohort','')) for i,s in enumerate(states)]
    csvout(output/'receptor_decisions.csv',rows)
    counts=summarize(rows,None)
    counts['scope']='UNIQUE_SOURCE_STATES_OF_WINNING_CALLS_ONLY; not the complete ensemble'
    dump(output/'counts.json',counts)
    dump(output/'retained_E_states.json',{'rule':RULE,'gate_recomputed':False,
                                        'states':[s for s in states if s['status']=='READY' and s.get('gate',{}).get('passed') is True]})

def main():
    parser=argparse.ArgumentParser(description='One-file original-state filtering and protein-aligned docking RMSD.')
    parser.add_argument('--source',type=Path,default=None,help='Optional Cell archive directory; default uses the bundled input_data.zip')
    parser.add_argument('--output',type=Path,default=None,help='New output directory; existing paths are never overwritten')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    source=args.source.resolve() if args.source else None
    archive=root.parent/'input_data.zip'
    if (source is not None and not source.is_dir()) or (source is None and not archive.is_file()):
        print('ERROR: Input archive not found. Provide --source PATH or restore input_data.zip.',file=sys.stderr)
        return 2
    output=args.output.resolve() if args.output else root/'outputs'/datetime.now().strftime('run_%Y%m%d_%H%M%S')
    if output.exists():
        print('ERROR: Output already exists; choose a new directory.',file=sys.stderr)
        return 2
    receipt={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
             'docking_performed':False,'source_modified':False,'implementation':'Single file; no scripts or examples directory required'}
    output.mkdir(parents=True)
    try:
        if source is None:
            receipt['input_archive_sha256']=core.sha256(archive)
            source=_extract_input_archive(archive,output/'input_data')
            receipt['source']=str(source)
        records,states,before=_source_records(source)
        detail=output/'details';detail.mkdir()
        dump(detail/'manifest.json',{'population_scope':'WINNING_CALLS_ONLY_NOT_FULL_SCENARIO','records':records})
        dump(detail/'receptor_states.json',{'states':states})
        _filter_original_states(states,detail/'receptor_filtering')
        result=evaluate(detail/'manifest.json',detail/'rmsd',state_manifest_path=detail/'receptor_states.json')
        summaries=readj(detail/'rmsd/summary.json')
        by_id={r['cell_id']:r for r in records}
        rows=[]
        for summary in summaries:
            cid=summary['cell_id']
            selected=readj(detail/'rmsd/selected'/cid/'all_pose_rmsd_min/selection.json')['pose']
            rows.append({'cell_id':cid,'RMSD_A':summary['all_pose_rmsd_min_A'],
                         'fixed_name_RMSD_A':selected['rmsd_fixed_name_A'],
                         'receptor_state':selected['state_id'],'seed':selected['docking_seed'],'mode':selected['mode'],
                         'vina_kcal_mol':selected['vina_kcal_mol'],'reaction_Fe_A':selected['reaction_Fe_A'],
                         'gate_passed':selected['gate_passed'],'Fe_score_hit':selected['Fe_score_hit'],
                         'expected_RMSD_A':by_id[cid]['selected_expected']['RMSD_symmetry_A']})
        csvout(output/'results.csv',rows)
        for path,digest in before.items():
            if core.sha256(path)!=digest:raise ValueError('Source changed during evaluation: '+path)
        receipt.update(status='PASS' if result['status'].startswith('PASS') else 'FAILED_OUTPUTS_PRESERVED',
                       cell_n=len(records),arm_pose_n=result['evaluated_arm_pose_n'],
                       physical_call_n=result['unique_physical_call_n'],unique_physical_pose_n=result['unique_physical_pose_n'],
                       source_file_n_verified=len(before),failed_expected_check_n=result['failed_expected_check_n'],
                       result_csv=str(output/'results.csv'),
                       scope='Reproduce supplied winning-call subset, not the full scenario pool',
                       receptor_filter='Original READY + strict gate=True; donor geometry is not recomputed')
    except Exception as error:
        receipt.update(status='FAILED_OUTPUTS_PRESERVED',error=type(error).__name__+': '+str(error))
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    dump(output/'receipt.json',receipt)
    if receipt['status']=='PASS':
        print('PASS: '+str(receipt['cell_n'])+' cells verified.')
        print('Results: '+receipt['result_csv'])
        return 0
    print('ERROR: See '+str(output/'receipt.json'),file=sys.stderr)
    return 2

if __name__=='__main__':raise SystemExit(main())
