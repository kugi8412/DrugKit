## Compare the smiles_processing pipeline with the original RDkit implementation

Download the `ESOL` dataset (1128 drug-like molecules):

```
curl -s --max-time 20 -L \
  "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv" \
  -o /tmp/esol.csv
```