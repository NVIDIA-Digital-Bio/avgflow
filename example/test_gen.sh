# python dit_pairwise_bias/generate.py \
#     -c config/generation_config/52m_gen.yaml \
#     -s "C#CCNC(=O)C1=C[C@@H](c2ccc(Br)cc2)C[C@@H](OCc2ccc(CO)cc2)O1" \
#     -n 64


python dit_pairwise_bias/generate_chembl.py \
    -c config/generation_config/52m_gen_chembl.yaml \
    -s /home/zhonglinc/storage/research/avgflow_release/dit_pairwise_bias/sampling/chembl_data_corrected.csv \
