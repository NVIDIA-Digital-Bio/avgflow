python avgflow/generate_from_smiles.py \
    -c config/generation_config/avgflow_52m_gen.yaml \
    -s "C#CCNC(=O)C1=C[C@@H](c2ccc(Br)cc2)C[C@@H](OCc2ccc(CO)cc2)O1" \
    -o ./example/output/gen_smiles \
    -n 40
