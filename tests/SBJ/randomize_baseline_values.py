# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
"""Randomize baseline design variable values within ±10% of their original values."""
import pathlib
import yaml
import numpy as np


def randomize_baselines(yaml_file, output_file=None, seed=42, perturbation=0.10):
    """
    Read a YAML config, randomize baseline values ±perturbation%, stay within [lb, ub].

    Args:
        yaml_file: Path to input YAML configuration
        output_file: Path to output YAML (if None, overwrites input)
        seed: Random seed for reproducibility
        perturbation: Perturbation fraction (default 0.10 = ±10%)
    """
    yaml_path = pathlib.Path(yaml_file)
    if output_file is None:
        output_file = yaml_path.with_stem(yaml_path.stem + "_perturbed")

    # Read YAML
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Modify baseline values in variables section
    variables = config.get("variables", {})
    for var_key, var_list in variables.items():
        # var_list format: [name, sp_index, link, coupling, lb, bl, ub, dim, var_type]
        if len(var_list) >= 7:
            lb, bl, ub = var_list[4], var_list[5], var_list[6]

            # Skip dummy variables or zero baselines
            if bl == 0.0 or ub == 0.0:
                continue

            # Random perturbation: sample uniformly from [-perturbation, +perturbation]
            alpha = rng.uniform(-perturbation, perturbation)
            new_bl = bl * (1.0 + alpha)

            # Clip to [lb, ub]
            new_bl = np.clip(new_bl, lb, ub)

            # Update baseline in-place
            var_list[5] = new_bl

            print(f"{var_key:4s} {var_list[0]:12s}: {bl:12.4f} -> {new_bl:12.4f} "
                  f"(perturb: {alpha*100:+6.2f}%, bounds: [{lb:.2e}, {ub:.2e}])")

    # Write modified YAML to output file
    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    yaml_file = pathlib.Path(__file__).parent / "SBJ.yaml"
    randomize_baselines(yaml_file, seed=42, perturbation=0.10)
