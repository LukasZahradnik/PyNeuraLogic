# Zebra Puzzle https://en.wikipedia.org/wiki/Zebra_Puzzle
from neuralogic.core import R, V, Template
from neuralogic.inference import InferenceEngine

color_vars = [V.Blue, V.Yellow, V.Ivory, V.Green, V.Red]
drink_vars = [V.OrangeJuice, V.Water, V.Tea, V.Milk, V.Coffee]
pet_vars = [V.Horse, V.Zebra, V.Fox, V.Dog, V.Snails]
smoke_vars = [V.Kools, V.Chesterfield, V.LuckyStrike, V.OldGold, V.Parliament]
person_vars = [V.Norwegian, V.Spaniard, V.Ukrainian, V.Japanese, V.Englishman]

template = Template()

# 1. There are five houses.
template += [R.house(i) for i in range(1, 6)]

template += R.next_to(V.X, V.Y) <= R.special.next(V.X, V.Y)  # Y is on the right side of X
template += R.next_to(V.X, V.Y) <= R.special.next(V.Y, V.X)  # Y is on the left side of X

template += R.solve(*person_vars, V.Zebra, V.Water) <= [
    *[R.house(var) for var in color_vars],
    *[R.house(var) for var in drink_vars],
    *[R.house(var) for var in pet_vars],
    *[R.house(var) for var in smoke_vars],
    *[R.house(var) for var in person_vars],
    # 2. The Englishman lives in the red house.
    R.special.eq(V.Englishman, V.Red),
    # 3. The Spaniard owns the dog.
    R.special.eq(V.Spaniard, V.Dog),
    # 4. Coffee is drunk in the green house.
    R.special.eq(V.Coffee, V.Green),
    # 5. The Ukrainian drinks tea.
    R.special.eq(V.Ukrainian, V.Tea),
    # 6. The green house is immediately to the right of the ivory house.
    R.special.next(V.Ivory, V.Green),
    # 7. The Old Gold smoker owns snails.
    R.special.eq(V.OldGold, V.Snails),
    # 8. Kools are smoked in the yellow house.
    R.special.eq(V.Kools, V.Yellow),
    # 9. Milk is drunk in the middle house.
    R.special.eq(V.Milk, 3),
    # 10. The Norwegian lives in the first house.
    R.special.eq(V.Norwegian, 1),
    # 11. The man who smokes Chesterfields lives in the house next to the man with the fox.
    R.next_to(V.Chesterfield, V.Fox),
    # 12. Kools are smoked in a house next to the house where the horse is kept.
    R.next_to(V.Kools, V.Horse),
    # 13. The Lucky Strike smoker drinks orange juice.
    R.special.eq(V.LuckyStrike, V.OrangeJuice),
    # 14. The Japanese smokes Parliaments.
    R.special.eq(V.Japanese, V.Parliament),
    # 15. The Norwegian lives next to the blue house.
    R.next_to(V.Norwegian, V.Blue),
    # In one house can live only one person, house can have only one color etc.
    R.special.alldiff(color_vars),
    R.special.alldiff(drink_vars),
    R.special.alldiff(pet_vars),
    R.special.alldiff(smoke_vars),
    R.special.alldiff(person_vars),
]

inferene_engine = InferenceEngine(template)

for solution in inferene_engine.query(R.solve(*person_vars, V.Zebra, V.Water)):
    print("Solution:", solution)
    for person in person_vars:
        if solution[person] == solution[V.Zebra]:
            print(f"Zebra owner is {person}")
        if solution[person] == solution[V.Water]:
            print(f"Water drinker is {person}")

# Solution: {'Norwegian': '1', 'Spaniard': '4', 'Ukrainian': '2', 'Japanese': '5', 'Englishman': '3', 'Zebra': '5', 'Water': '1'}
# Water drinker is Norwegian
# Zebra owner is Japanese
