# AP102 Project

## How to run

If the default python interpreter is python 3, then
```bash
python simulation.py --map name_of_map
```

else 

```bash
python3 simulation.py --map name_of_map
```

Specifically, if you want to simulate the map `circuit 3`, type
```bash
python simulation.py --map "circuit 3"
```
If any dependency error occurs, please install missing packages.

## How to add new map
Create two images. One for the static obstacles and one for dynamic obstacles. Then fix start and goal positions.
Update the information in `config.toml` file.