## Rule metadata

```python
from neuralogic.model import Var, Term, Atom, Template, Metadata, Activation, Aggregation
from neuralogic.model.java_objects import use_java_factory
from neuralogic.settings import Settings

with use_java_factory(Settings()):

    rule = (Atom.embed(Var.A)[3, 1] <= Atom.c_26(Var.A)) | Metadata(activation=Activation.TANH, learnable=True)
    # {3, 1} embed(A) :- c_26(A). [learnable=true, activation=tanh]
```

## Predicate access

```python
    predicate = Atom.embed/2
    # embed/2

    print(Atom.embed/4)  # embed/4
    print(Atom.embed/0)  # embed/0
    print((Atom.embed/8).arity)  # 8
    print(Atom.special.special_embed/8)  # @special_embed/8
```

## Predicate metadata

```python
    Atom.embed/2 | Metadata(activation=Activation.LUKASIEWICZ)
    # embed/2 [activation=lukasiewicz]
```


## Template

```python

with use_java_factory(Settings()):
    template = Template()

    template.add_rule(Atom.xor[1, 8] <= Atom.xy[8, 2])

    template.add_examples([
        Atom.xor[0] <= Atom.xy[[0, 0]],
        Atom.xor[1] <= Atom.xy[[0, 1]],
        Atom.xor[1] <= Atom.xy[[1, 0]],
        Atom.xor[0] <= Atom.xy[[1, 1]],
    ])

    samples, weights = template.build()
```
