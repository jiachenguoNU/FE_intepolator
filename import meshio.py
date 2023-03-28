import meshio

# two triangles and one quad
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 4, 3]]),
]

mesh = meshio.Mesh(
    points,
    cells,
)
meshio.write_points_cells("foo.vtk", points, cells)