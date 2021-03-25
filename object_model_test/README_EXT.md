```python
from object_model_test.model import Atom, Var, Term

X = Var.X
Y = Var.Y
```

### Terms
```python
atom_with_term = Atom.shape(Var.X, Term.circle)
# shape(X, circle).

atom_with_term = Atom.shape(Var.x, Term.cIrcLE)
# shape(X, circle).
```

### Special and Private

```python
test = Atom.private.private_predicate
# *private_predicate.

test = Atom.special.special_predicate
# @special_predicate.

test = Atom.special.private.predicate
# *@predicate.

test_2 = test(Var.X, Var.Y)
# *@predicate(X, Y).

test_3 = test(Var.Y, Var.Z)
# *@predicate(Y, Z).
```

### Sharing predicate instance

```python
print(test_3.predicate == test_2.predicate)  # True
print(test.predicate == test_2.predicate)  # False

print(test.predicate)  # *@predicate/0
print(test_2.predicate)  # *@predicate/2
print(test_3.predicate)  # *@predicate/2

print(test.predicate.name, test.predicate.arity, test.predicate.private, test.predicate.special)
# predicate 0 True True
```

### Negation
```python
negated = -Atom.positive
# ~positive.

negated = ~Atom.positive
# ~positive.

negated = -Atom.positive(Var.X)[1, 2]
# {1, 2} ~positive(X).

negated = ~Atom.positive[1, 2]
# {1, 2} ~positive.
```


### Writing rules (no changes, just more features)
```python
rule = test_2 <= test_3
# *@predicate(X, Y) :- *@predicate(Y, Z).

rule = test_2 <= -test_3
# *@predicate(X, Y) :- ~*@predicate(Y, Z).
```

### Fixed, dimensions, vectors, scalars
```python
rule = Atom.abc[1] <= (Atom.xyz[1, 2], Atom.efg[[1, 2]])
# 1 abc :- {1, 2} xyz, [1, 2] efg.

fixed_rule = Atom.abc[1].fixed() <= (Atom.xyz[1, 2].fixed(), Atom.efg[[1, 2]].fixed())
# <1> abc :- <{1, 2}> xyz, <[1, 2]> efg.
```

New change:
```python
dimension = Atom.abc[1,]
# {1} abc.a
not_dimension = Atom.abc[1]
# 1 abc.
```

### Vectorized xor example
```python
template = Atom.xor[1, 8] <= Atom.xy[8,2]
#  {1, 8} xor :- {8, 2} xy.

examples = [
    Atom.xor[0] <= Atom.xy[[0, 0]],
    Atom.xor[1] <= Atom.xy[[0, 1]],
    Atom.xor[1] <= Atom.xy[[1, 0]],
    Atom.xor[0] <= Atom.xy[[1, 1]],
]

#  0 xor :- [0, 0] xy.
#  1 xor :- [0, 1] xy.
#  1 xor :- [1, 0] xy.
#  0 xor :- [1, 1] xy.
```


### Better recycling
```python
first = Atom.one(Var.X)

two = first[1]
three = first[2]

print(first)
print(two)
print(three)

# one(X).
# 1 one(X).
# 2 one(X).
```
