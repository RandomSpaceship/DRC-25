import numpy as np

tree = {
    0: [(1, 10), (2, 20)],
    2: [(3, 40)],
}

a = [[(v1[0], k) for v1 in v] for k, v in tree.items()]
inverse_tree = {}
for k, values in tree.items():
    for prev, _ in values:
        inverse_tree[prev] = k

print(a)
print(inverse_tree)

arr = [1, 2, 3]
arr2 = [7, 8, 9]

arr.extend(arr2)
print(arr)
