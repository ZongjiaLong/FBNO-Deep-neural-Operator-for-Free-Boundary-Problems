import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging

"""
Stefan Problem Simulation with Randomized Parameters
Simulates phase change dynamics with heat diffusion and moving boundary conditions.

Under the current parameter configuration, certain operating conditions may not
be calculated correctly due to:
1. Potential numerical stability violations (CFL condition)
2. Boundary condition implementation constraints
3. Physical parameter range limitations

Users are STRONGLY ADVISED to:
1. Visually inspect all generated plots for unphysical artifacts
2. Verify interface position behavior remains physically plausible
3. Check temperature distributions maintain expected profiles

Particular attention should be paid to:
- Sudden jumps in interface position
- Temperature values exceeding physical bounds
- Non-monotonic behavior in cooling phases

Please validate all results before drawing scientific conclusions.
"""




"""
# ==============================================
# CONFIGURATION SECTION - EDIT THESE PATHS AS NEEDED
# ==============================================
"""
BASE_OUTPUT_FOLDER = ".\\stefan_data"  # Base output directory
VISUALISATION_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "visualisation")  # For plots
DATA_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "data")  # For .npy files


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PhysicalParameters:
    """Container for physical constants and parameters"""
    thermal_diffusivity: float = 1.15e-6  # m²/s
    thermal_conductivity: float = 2.18  # W/m·K
    density: float = 917.0  # kg/m³
    latent_heat: float = 334000.0  # J/kg
    melting_temp: float = 0.0  # °C
    initial_temp: float = 3.0  # °C

    @property
    def stefan_coefficient(self) -> float:
        """Calculate Stefan condition coefficient"""
        return self.thermal_conductivity / (self.density * self.latent_heat)


@dataclass
class SimulationParameters:
    """Container for simulation parameters"""
    spatial_points: int = 101
    time_steps: int = 5001
    total_time: float = 3600.0  # s
    max_length: float = 2.5  # m
    max_temp: float = 20.0  # °C
    cooling_magnitude: float = 6e-4
    heat_source_min: float = 0.0005
    heat_source_max: float = 0.0020
    period_min: float = 1.8
    period_max: float = 6.0
    heat_period: float = 100.0
    num_simulations: int = 10
    random_seed: int = 42
    output_folder: str = VISUALISATION_FOLDER
    data_folder: str = DATA_FOLDER

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        logger.info(f"Created output folders:\n"
                    f"- Visualisations: {self.output_folder}\n"
                    f"- Data: {self.data_folder}")




class StefanProblemSimulator:
    """
    Simulates the Stefan problem with randomized heat source parameters

    Attributes:
        phys_params (PhysicalParameters): Physical constants
        sim_params (SimulationParameters): Simulation settings
        results (List[SimulationResult]): Storage for simulation outputs
    """

    def __init__(self, phys_params: PhysicalParameters, sim_params: SimulationParameters):
        self.phys_params = phys_params
        self.sim_params = sim_params
        self.results: List[dict] = []

        # Calculate derived parameters
        self.adjusted_diffusivity = self.phys_params.thermal_diffusivity * 2 * 15
        self.stefan_coefficient = self.phys_params.stefan_coefficient * 10000

        # Normalized parameters
        self.norm_diffusivity = (self.adjusted_diffusivity /
                                 (self.sim_params.max_length ** 2) *
                                 self.sim_params.total_time)

        self.norm_stefan = (self.stefan_coefficient /
                            (self.sim_params.max_length ** 2) *
                            self.sim_params.total_time *
                            self.sim_params.max_temp)

        self.norm_cooling = (self.sim_params.cooling_magnitude *
                             self.sim_params.total_time /
                             self.sim_params.max_length)

    def boundary_condition(self, x: float, t: float) -> float:
        """Boundary condition at x=0"""
        return self.phys_params.initial_temp / self.sim_params.max_temp

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """Initial temperature distribution"""
        return (self.phys_params.initial_temp / self.sim_params.max_temp *
                (1.0 - x)** 2)

    def cooling_function(self, t: float, period: float) -> float:
        """Time-dependent cooling function"""
        return (self.norm_cooling *
                (np.sin(t * 0.00698 / period * self.sim_params.total_time - np.pi / 2) + 1))

    def heat_source(self, x: float, t: float) -> float:
        """Spatial heat source function"""
        return (self.current_heat_magnitude *
                (np.sin(3.14 / self.sim_params.heat_period * x * self.sim_params.max_length) + 1))

    def check_stability(self, r: float) -> None:
        """Check numerical stability condition"""
        if r > 0.5:
            logger.warning(f"Stability condition violated: r = {r:.3f} > 0.5")

    def run_single_simulation(self, heat_magnitude: float, period: float) -> dict:
        """Execute one complete simulation with given parameters"""
        # Initialize grids
        dx = 1.0 / (self.sim_params.spatial_points - 1)
        dt = 1.0 / (self.sim_params.time_steps - 1)
        r = self.norm_diffusivity * dt / dx ** 2
        self.check_stability(r)

        xi = np.linspace(0, 1.0, self.sim_params.spatial_points)
        t_norm = np.linspace(0, 1.0, self.sim_params.time_steps)

        # Initialize fields
        u = np.zeros((self.sim_params.spatial_points, self.sim_params.time_steps))
        s = np.zeros(self.sim_params.time_steps)
        s[0] = 1.0 / self.sim_params.max_length
        u[:, 0] = self.initial_condition(xi)
        u[-1, 0] = 0

        # Time integration
        for n in range(self.sim_params.time_steps - 1):
            # Update interface position
            interface_flux = (-self.norm_stefan / s[n] *
                              (u[-1, n] - u[-2, n]) / dx -
                              self.cooling_function(t_norm[n], period))
            s[n + 1] = s[n] + interface_flux * dt

            # Update temperature field
            u[0, n + 1] = self.boundary_condition(0, t_norm[n + 1])
            u[-1, n + 1] = 0

            for i in range(1, self.sim_params.spatial_points - 1):
                advection = (xi[i] / s[n] * ((s[n + 1] - s[n]) / dt) *
                             (u[i + 1, n] - u[i, n]) / dx)
                diffusion = (self.norm_diffusivity / (s[n] ** 2) *
                             (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n]) / dx ** 2)
                u[i, n + 1] = u[i, n] + dt * (advection + diffusion +
                                              self.heat_source(xi[i] * s[n], t_norm[n]))

            # Enforce boundary conditions
            u[-1, n + 1] = 0
            u[0, n + 1] = self.boundary_condition(0, t_norm[n + 1])

        # Calculate physical coordinates
        x_norm = xi.reshape(-1, 1) * s.reshape(1, -1)

        return {
            'heat_magnitude': heat_magnitude,
            'period': period,
            'temperature_field': u,
            'interface_position': s,
            'normalized_space': xi,
            'physical_space': x_norm,
            'normalized_time': t_norm
        }

    def visualize_result(self, result: dict, sim_num: int) -> None:
        """Generate and save visualization plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Normalized coordinates plot
        im1 = ax1.imshow(result['temperature_field'],
                         extent=[0, 1.0, 0, 1.0],
                         aspect='auto', cmap='turbo', origin='lower')
        fig.colorbar(im1, ax=ax1, label='Temperature')
        ax1.set(xlabel='Time (t)', ylabel='Position (x)',
                title=f'Temperature Distribution\nq={result["heat_magnitude"] * 1000:.3f}, p={result["period"]:.1f}')

        # Physical coordinates plot
        t_real = result['normalized_time'] * self.sim_params.total_time
        x_real = result['normalized_space'].reshape(-1, 1) * (
                result['interface_position'] * self.sim_params.max_length).reshape(1, -1)
        u_real = result['temperature_field'] * self.sim_params.max_temp

        im2 = ax2.pcolormesh(t_real, x_real, u_real, cmap='turbo', shading='auto')
        fig.colorbar(im2, ax=ax2, label='Temperature (°C)')
        ax2.set(xlabel='Time (s)', ylabel='Position (m)',
                title='Temperature Distribution (Physical)')

        plt.tight_layout()
        plot_path = os.path.join(self.sim_params.output_folder, f'stefan_sim_{sim_num + 1}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualization to: {plot_path}")

    def run_all_simulations(self) -> None:
        """Execute multiple simulations with randomized parameters"""
        np.random.seed(self.sim_params.random_seed)

        for sim_num in tqdm(range(self.sim_params.num_simulations), desc="Running simulations"):
            # Randomize parameters
            heat_mag = np.random.uniform(self.sim_params.heat_source_min,
                                         self.sim_params.heat_source_max)
            period = np.random.uniform(self.sim_params.period_min,
                                       self.sim_params.period_max)

            # Convert to normalized magnitude
            self.current_heat_magnitude = (heat_mag *
                                           self.sim_params.total_time /
                                           self.sim_params.max_temp)

            # Run simulation
            result = self.run_single_simulation(self.current_heat_magnitude, period)
            self.results.append(result)

            # Visualize and save
            self.visualize_result(result, sim_num)

        # Save all results
        output_path = os.path.join(self.sim_params.data_folder, "stefan_data.npy")
        np.save(output_path, self.results)
        logger.info(f"\nCompleted {self.sim_params.num_simulations} simulations.\n"
                    f"Visualizations saved to: {self.sim_params.output_folder}\n"
                    f"Data saved to: {output_path}")

def main():
    """Main execution function"""
    # Initialize parameters
    phys_params = PhysicalParameters()
    sim_params = SimulationParameters()

    # Create and run simulator
    simulator = StefanProblemSimulator(phys_params, sim_params)
    simulator.run_all_simulations()


# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm


# Recreate the necessary classes and functions from your original code
@dataclass
class PhysicalParameters:
    thermal_diffusivity: float = 1.15e-6
    thermal_conductivity: float = 2.18
    density: float = 917.0
    latent_heat: float = 334000.0
    melting_temp: float = 0.0
    initial_temp: float = 3.0


@dataclass
class SimulationParameters:
    spatial_points: int = 101
    time_steps: int = 5001
    total_time: float = 3600.0
    max_length: float = 2.5
    max_temp: float = 20.0
    cooling_magnitude: float = 6e-4
    output_folder: str = "D:\\desktop\\stefan_plots11\\visualisation"
    data_folder: str = "D:\\desktop\\stefan_plots11\\data"


class ResultVisualizer:
    def __init__(self, phys_params: PhysicalParameters, sim_params: SimulationParameters):
        self.phys_params = phys_params
        self.sim_params = sim_params

    def visualize_result(self, result: Dict, sim_num: int) -> None:
        """Generate and save visualization plots for a single result"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 1.25),
                                       gridspec_kw={'wspace': 0.2})  # Add spacing between subplots

        # Normalized coordinates plot
        im1 = ax1.imshow(result['temperature_field'],
                         extent=[0, 1.0, 0, 1.0],
                         aspect='auto', cmap='turbo', origin='lower')
        ax1.axis('off')  # Turn off axis

        # Physical coordinates plot
        t_real = result['normalized_time'] * self.sim_params.total_time
        x_real = result['normalized_space'].reshape(-1, 1) * (
                result['interface_position'] * self.sim_params.max_length).reshape(1, -1)
        u_real = result['temperature_field'] * self.sim_params.max_temp

        im2 = ax2.pcolormesh(t_real, x_real, u_real, cmap='turbo', shading='auto')
        ax2.axis('off')  # Turn off axis

        plt.tight_layout(pad=0)  # Remove padding between subplots
        plot_path = os.path.join(self.sim_params.output_folder, f'stefan_sim_{sim_num + 1}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved visualization to: {plot_path}")


def load_and_visualize():
    # Initialize parameters (must match original simulation)
    phys_params = PhysicalParameters()
    sim_params = SimulationParameters()

    # Create visualizer
    visualizer = ResultVisualizer(phys_params, sim_params)

    # Load the .npy file
    data_path = os.path.join(sim_params.data_folder, "stefan_data.npy")
    results = np.load(data_path, allow_pickle=True)

    # Visualize each result
    for i, result in enumerate(tqdm(results, desc="Visualizing results")):
        visualizer.visualize_result(result, i)

    print(f"\nSuccessfully visualized {len(results)} simulations")


if __name__ == "__main__":
    load_and_visualize()