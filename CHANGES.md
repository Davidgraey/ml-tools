0.1.3
=====================
Aligned BasalModel with the Layer ABC and added differentiable clustering layers.
    - check examples/NNet_clustering_layers_example.py and the app
    - types BasalModel: subclass registry, serialization and matching Layers more closely
    - removed BasalModel params
    - models/layers/clustering_layers.py: CentroidLayer, PLSOMLayer, GPLSOMLayer, FreePLSOMLayer -- soft-assignment prototype layers with their own clustering energy
    - examples/NNet_clustering_layers_example.py

0.1.2
=====================
Modified our layers -- check examples.
    - Modified clustering mechanisms to work as Network Layers -- need to continue work here
    - added text and MLM examples
    - added Hyena implementation 

0.1.1
=====================
Reformatted several key layers (MoE) and Decision System-One. check examples.
    - models/layers Spectre layers now truer-to-paper (individual heads, not a shared represetnation)
    - models/layers MoE router now has routing, bias and factor included
    - models/layers DecisionHead extended with MOE trunks
    - models/layers added in the GatherLayer and other token / attention mask / target mask behaviors for Language models
    - encoders/ Text Encoder and tokenizers extended with additional training tasks (BART, Bert, electra)
- TODO: add Hyena / H3 FFT as an option. 

0.1.0
=====================
Renamed the package from ml_tools to polyergalio and prepared for PyPI
publication:
    - src/ml_tools became src/polyergalio, all imports updated
    - pyproject.toml: dynamic version resolution, explicit src-layout
      package discovery
    - removed import-time logging

0.0.1
=====================
Version algorithm aggregations - serialization and unserialize functions added
    - Clusterings, SOM, PLSOM, free-SOM and CentNN modified
    - Classification models updated with 
    - NNet layers, network and DAG
    - Generator "shapes patterns" added for clustering and images

0.0.0 
=====================
Version zero - collecting algorithms and tools from my other repos and 
projects.
