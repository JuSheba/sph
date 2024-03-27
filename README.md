# Smoothed Particle Hydrodynamics

### Realisation of smoothed particle hydrodynamics for a ring of particles around a massive body.

- To use this module please run the following commands:

`pip install --upgrade pip`

`pip install numpy`

`pip install -U scikit-learn`

`pip install -U matplotlib`

`pip install argparse`

`pip install json`

- You can change the model parameters in *param_conf.json*. For example: the number of particles, the massive body mass or the time frame.

- The main module is *sph.py*. It calculates the coords array for every time point and writes it to a txt file.
- You can use *draw.py* to generate an animation of this data.
