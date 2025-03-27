from schrodinger import structure

# Load structures with the full path to the input file
structures = structure.StructureReader(r"C:\Users\kmml910\OneDrive - AZCollaboration\Documents\Coding\Mouse_microsomal_stability\ML ready dataset\ligprep_training_dataset_578compounds\adme_Calculation\adme_Calculation-out.maegz")

# Dictionary to store lowest energy conformer
conformer_dict = {}

for st in structures:
    
# Debug: Print available properties using keys

    # print(f"Available properties for structure: {list(st.property.keys())}") 
    

    mol_id = st.property.get('s_m_title')  # Adjust based on actual property key
    energy = st.property.get('r_mmod_Potential_Energy-S-OPLS')  # Adjust based on actual property key

    # Ensure keys exist
    if mol_id is None or energy is None:
        print("Missing required properties, skipping...")
        continue

    # Update dictionary to only keep lowest energy conformer
    if mol_id not in conformer_dict or energy < conformer_dict[mol_id][1]:
        conformer_dict[mol_id] = (st, energy)

# Write out the lowest energy conformers
with structure.StructureWriter('single_conformers_validation.mae') as writer:
    for conformer, _ in conformer_dict.values():
        writer.append(conformer)
        
with structure.StructureWriter('single_conformers_validation.sdf') as writer:
    for conformer, _ in conformer_dict.values():
        writer.append(conformer)
        
with structure.StructureWriter('single_conformers_validation.csv') as writer:
    for conformer, _ in conformer_dict.values():
        writer.append(conformer)
