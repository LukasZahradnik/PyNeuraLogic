## Writing templates via objects (proposal)

Writing rules in form of

- no body atom (facts)
```
head_atom_name(terms)[weights]
```

- one body atom
```
head_atom_name(terms)[weights] > body_atom_name(terms)[weights]
```

- multiple body atoms
```
head_atom_name(terms)[weights] > (body_atom_name1(terms)[weights], body_atom_name2(terms)[weights])
```



## Basic usage

```python
from model import Var, Atom

print(Atom.predict)
# >> predict.

print(Atom.predict[1])
# >> {1} predict.

print(Atom.predict(Var.X))
# >> predict(X).

print(Atom.predict(Var.X)[1, 2])
# >> {1, 2} predict(X).


# Also possible..
predict = Atom.predict

print(predict(Var.X))
# >> predict(X).

print(predict)
# >> predict.


# Rules with bodies:

print(Atom.direction(Var.X) > Atom.train(Var.X)[1])
# >>  direction(X) :- {1} train(X).

# Difference between:
rule = [(Atom.direction(Var.X) > Atom.train(i)[1]) for i in [1, 2]]  # Two rules with one body atom
print([str(r) for r in rule])
# and
rule = [Atom.direction(Var.X) > [Atom.train(i)[1] for i in [1, 2]]]  # One rule with two body atoms
print([str(r) for r in rule])

# >> ['direction(X) :- {1} train(1).', 'direction(X) :- {1} train(2).']
# >> ['direction(X) :- {1} train(1), {1} train(2).']
```


### Example of trains template

```python
from model import Var, Atom


shapes = ['ellipse', 'rectangle', 'bucket', 'hexagon', 'u_shaped']
roofs = ['jagged', 'arc', 'none', 'flat', 'peaked']
loadshapes = ['hexagon', 'triangle', 'diamond', 'rectangle', 'circle']
vagon_atoms = [Atom.shape, Atom.length, Atom.sides, Atom.wheels, Atom.loadnum, Atom.loadshape, Atom.roof]

X = Var.X
Y = Var.Y

template = [
    *[Atom.shape(X, Y) > Atom.shape(X, Y, s)[1] for s in shapes],
    *[Atom.length(X, Y) > Atom.length(X, Y, s)[1] for s in ['short', 'long']],
    *[Atom.sides(X, Y) > Atom.sides(X, Y, s)[1] for s in ['not_double', 'double']],
    *[Atom.roof(X, Y) > Atom.roof(X, Y, s)[1] for s in roofs],
    *[Atom.wheels(X, Y) > Atom.wheels(X, Y, s)[1] for s in [2, 3]],
    *[Atom.loadnum(X, Y) > Atom.loadnum(X, Y, s)[1] for s in [0, 1, 2, 3]],
    *[Atom.loadshape(X, Y) > Atom.loadshape(X, Y, s)[1] for s in loadshapes],
    Atom.vagon(X, Y) > (atom(X, Y)[1] for atom in vagon_atoms),
    *[Atom.train(X) > Atom.vagon(X, i)[1] for i in [1, 2, 3, 4]],
    Atom.direction(X) > Atom.train(X)[1],
]

# Template as string
template = "\n".join(str(t) for t in template)

print(template)
```

output:
```
shape(X, Y) :- {1} shape(X, Y, ellipse).
shape(X, Y) :- {1} shape(X, Y, rectangle).
shape(X, Y) :- {1} shape(X, Y, bucket).
shape(X, Y) :- {1} shape(X, Y, hexagon).
shape(X, Y) :- {1} shape(X, Y, u_shaped).
length(X, Y) :- {1} length(X, Y, short).
length(X, Y) :- {1} length(X, Y, long).
sides(X, Y) :- {1} sides(X, Y, not_double).
sides(X, Y) :- {1} sides(X, Y, double).
roof(X, Y) :- {1} roof(X, Y, jagged).
roof(X, Y) :- {1} roof(X, Y, arc).
roof(X, Y) :- {1} roof(X, Y, none).
roof(X, Y) :- {1} roof(X, Y, flat).
roof(X, Y) :- {1} roof(X, Y, peaked).
wheels(X, Y) :- {1} wheels(X, Y, 2).
wheels(X, Y) :- {1} wheels(X, Y, 3).
loadnum(X, Y) :- {1} loadnum(X, Y, 0).
loadnum(X, Y) :- {1} loadnum(X, Y, 1).
loadnum(X, Y) :- {1} loadnum(X, Y, 2).
loadnum(X, Y) :- {1} loadnum(X, Y, 3).
loadshape(X, Y) :- {1} loadshape(X, Y, hexagon).
loadshape(X, Y) :- {1} loadshape(X, Y, triangle).
loadshape(X, Y) :- {1} loadshape(X, Y, diamond).
loadshape(X, Y) :- {1} loadshape(X, Y, rectangle).
loadshape(X, Y) :- {1} loadshape(X, Y, circle).
vagon(X, Y) :- {1} shape(X, Y), {1} length(X, Y), {1} sides(X, Y), {1} wheels(X, Y), {1} loadnum(X, Y), {1} laodshape(X, Y), {1} roof(X, Y).
train(X) :- {1} vagon(X, 1).
train(X) :- {1} vagon(X, 2).
train(X) :- {1} vagon(X, 3).
train(X) :- {1} vagon(X, 4).
direction(X) :- {1} train(X).
```
