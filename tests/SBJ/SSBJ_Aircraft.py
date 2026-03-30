from typing import List

import numpy as np
import math

class SSBJ_Aircraft:
    """Class representing a Supersonic Business Jet (SSBJ) aircraft with performance calculations."""

    def __init__(self, h=55000, Mach=1.4, SFCp=2, We=15000, LDr=5.0, Ws=25000, Wf=25000):
        """Initialize the SSBJ aircraft with given parameters.

        Args:
            h (float): Altitude in feet
            Mach (float): Mach number
            SFCp (float): Specific fuel consumption (per hour)
            We (float): Engine weight in pounds
            LDr (float): Lift-to-drag ratio
            Ws (float): Structural weight in pounds
            Wf (float): Fuel weight in pounds
        """
        self.h = h
        self.Mach = Mach
        self.SFCp = SFCp
        self.We = We
        self.LDr = LDr
        self.Ws = Ws
        self.Wf = Wf
        
        # Calculate total weight
        

    def _calculate_theta_r(self) -> float:
        """Calculate the temperature ratio based on altitude.

        Returns:
            float: Temperature ratio theta_r
        """
        if self.h < 36089:
            return 1 - 0.000006875 * self.h
        else:
            return 0.7519

    def _calculate_range(self, Wt, theta_r) -> float:
        """Calculate the aircraft range using the Breguet range equation.

        Returns:
            float: Range in nautical miles
        """
        return (self.Mach * self.LDr * 661.0 * 
                np.sqrt(theta_r / self.SFCp) * 
                math.log(Wt / (Wt - self.Wf)))

    def _calculate_constraints(self, range) -> List[float]:
        """Calculate the constraint value for the optimization problem.

        Returns:
            float: Constraint value g
        """
        return [-range / 2000.0 + 1.0]

    def get_results(self) -> dict:
        """Get all calculated results as a dictionary.

        Returns:
            dict: Dictionary containing all calculated values
        """
        return {
            "altitude": self.h,
            "Mach": self.Mach,
            "SFCp": self.SFCp,
            "engine_weight": self.We,
            "lift_drag_ratio": self.LDr,
            "structural_weight": self.Ws,
            "fuel_weight": self.Wf,
            "total_weight": self.SBJ_aircraft_analysis()[0],
            "temperature_ratio": self._calculate_theta_r(),
            "range": self.SBJ_aircraft_analysis()[1],
            "constraint": self._calculate_constraints(range=self.SBJ_aircraft_analysis()[1])
        }

    def __str__(self) -> str:
        """String representation of the aircraft.

        Returns:
            str: Formatted string with key parameters
        """
        return (f"SSBJ Aircraft (h={self.h} ft, Mach={self.Mach}, Wt={self.Wt} lb, "
                f"range={self.range:.1f} nm, violation={sum(self.g):.3f})")

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            str: Detailed string representation
        """
        return (f"SSBJ_Aircraft(h={self.h}, Mach={self.Mach}, SFCp={self.SFCp}, "
                f"We={self.We}, LDr={self.LDr}, Ws={self.Ws}, Wf={self.Wf})")

    def SBJ_aircraft_analysis(self):
        Wt = self.We + self.Wf + self.Ws
        
        # Calculate theta_r based on altitude
        theta_r = self._calculate_theta_r()
        
        # Calculate range
        range = self._calculate_range(Wt=Wt, theta_r = theta_r)
        
        return [Wt, range]
    
    def SBJ_aircraft_opt(self, Wt, range):
        return [Wt, self._calculate_constraints(range=range)]

# Example usage and testing
if __name__ == "__main__":
    # Create an instance with default values
    aircraft = SSBJ_Aircraft()
    
    # Print results
    # print(aircraft)
    # print(f"Range: {aircraft.range:.1f} nautical miles")
    # print(f"Constraint violation: {sum(aircraft.g):.3f}")
    
    # Get all results as dictionary
    results = aircraft.get_results()
    for key, value in results.items():
        print(f"{key}: {value}")