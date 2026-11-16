These are snapshots of the pre-refactor
`solutions/my_drone_basic_control.py` and `solutions/my_drone_map_building.py`.
They preserve the original algorithms as independent regression oracles.
Do not update them to follow changes in the component implementation.

The pre-existing untracked `solutions/drone.py` was identical to the mapper
snapshot. Its methods now live in the solution components; its original source
is also preserved here. The old solution paths remain as wiring wrappers.

TODO: delete later when refactor approved
