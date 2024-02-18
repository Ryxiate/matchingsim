# Matching Simulations in Geo 3D-SR

This repository contains a Python implementation of several matching simulation for Geometric 3D-SR problem and their respective evaluations, and is a project by SJTU students.

## Introduction

The study of Geometric 3D-SR problem has practical significance as the features used in matching can be from surveys conducted concerning lifestyle and backgrounds of students. These features provide an informative indicator of how students may evaluate others. Generally, it is desired for students with more similar features to be matched together. Therefore, matching derived from features reported by students can save the arduous and difficult process of collecting students utilities towards others, while is assumed to give stable and satisfactory results.

## Usage

Follow the following steps to run the simulation process.

1. Clone the repository:

```bash
git clone https://github.com/Ryxiate/matchingsim.git
```

2. \[Optional\] Open the `config.json` file to tweak the simulation parameters.

3. Setup the virtua env:

```bash
python3 -m venv .venv

# For Linux users
source .venv/bin/activate
# For Windows users
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

4. Run the algorithm by executing the main Python script

```bash
python 'main.py'
```

5. The figures will be shown in a different window or saved directly to `./figure/` depending on your configuration.

## Configuration

The following are explanations on the configuration parameters in `config.json`:

### SimulationOptions
__steps__: list, can contain 1 to 4.
- Step 1: run all algorithms with the default data settings. Plot the average utilities graph, the individual utilities graph and histogram.
- Step 2: run all algorithms with different noise levels.
- Step 3: run all algorithms with one-directional bias and indifference present individually, neither, or both.
- Step 4: runn all algorithms with the default data settings and evaluate the performances with the selected evaluation technique (`eval_list`)

__iter_num__: iteration number for Monte Carlo method

__room_nums__: list of different number of rooms to run simulations for all steps.

__noise_list__: list of different level of noise to run simulations in step 2.

__init_noise__: the default noise level to run simulations for all steps except step 2.

__solver_list__: list of matching algorithms to use in simulations. The available options:
```json
[
    "serial_dictatorship", 
    "random_serial_dictatorship",
    "match_by_characteristics", 
    "SD_by_rooms", 
    "random_match"
]
```
__eval_list__: list of evaluation techniques to use in step 4. The available options:
```json
[
    "preference_swapper", 
    "ttc_evaluator"
]
```
__pref_dist__: method for preferences to generate. The available options:
- "geo": number of preferences will follow geometric distribution with a limit linearly related with the number of rooms
- "uniform": number of preferences will be uniformly distributed from 1 to the limit linearly related with the number of rooms

### GraphOptions
__(rooms_n_rows, rooms_n_cols)__: arrangements for subplots indexed by different room numbers. Must satisfy rooms_n_rows * rooms_n_cols = len(room_nums)

__(noise_n_rows, noise_n_cols)__: arrangements for subplots indexed by different noise levels. Must satisfy noise_n_rows * noise_n_cols = len(noise_list)

__save_not_show__: boolean value to determine if the figure will be shown in a different window or directly saved.

__use_title__: boolean value to determine if the figures will have suptitles.

__dpi__: dpi for the generated figures.

__small_figure_ratio__: figure size ratio for figures without subplots.

__large_figure_ratio__: figure size ratio for figures with subplots.