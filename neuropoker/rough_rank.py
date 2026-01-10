from stats import DECK as deck
#hardcode some stuff so that checking ranks is easier

# high card hands = 52
print(len(deck))
# pair hands =
pair = {frozenset([a, b]) for a in deck for b in deck if a[1] != b[1] and a[0]==b[0]}
print(len(pair))

#2pair
twopair = {
    frozenset([a, b, c, d])
              for a in deck
              for b in deck
              for c in deck
              for d in deck
              if frozenset([a, b]) in pair and
              frozenset([c, d]) in pair and
              a[0] != c[0]
    }

#3ok

three = {
    frozenset([a, b, c])
    for a in deck
    for b in deck
    for c in deck
    if len(set([a[0], b[0], c[0]]))==1 and
        len(set([a[1], b[1], c[1]]))==3
              }

print(len(three))
