- Dataset (Used to be contained inside of `Problem`)
    - Should be equivalent to torch's Dataset
    > - Values y (labels, targets) and x
    > - Examples and queries file
    > - Example and query lists of python objects


- BuiltDataset (Used to be `Dataset`)
    - No equivalent
    - Grounded and neuralized Dataset


- Template (Used to be `Problem`)
    - No equivalent
    - Contains rules - definition of model (parameters, architecture)
    - Handles Dataset building into BuiltDataset and template building into NeuraLogic (Model)


- NeuraLogic (Model)
    - Provides interface for the forward propagation
    - Should be equivalent to torch's modules inherited from the `nn.Module`

- Evaluator
    - No direct equivalent
    - Ignite calls it trainer, evaluator, engine etc.
