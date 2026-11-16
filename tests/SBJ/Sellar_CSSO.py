import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Sellar Problem Disciplines ---
class SellarDiscipline1: # Structure
    def __init__(self):
        self.x1 = 0.0 # Local design variable for Disc1
        self.x2_shared = 0.0 # Shared design variable (input to Disc1)
        self.y2_in = 100.0 # Linking variable input from Disc2
        self.y1_out = 10.0 # Linking variable output to Disc2

    def evaluate(self, x2_in, y2_in):
        # Update inputs
        self.x2_shared = x2_in
        self.y2_in = y2_in
        # Calculate output y1
        self.y1_out = self.x1**2 + self.x2_shared + self.y2_in
        return self.y1_out

    def optimize(self, x2_in, y2_in):
        # In a distributed context, Disc1 might optimize its local variable x1
        # to satisfy some local objective or to contribute to the global objective.
        # Here, let's assume it tries to minimize its contribution to the global objective
        # while keeping y1 consistent. This is a simplification for illustration.

        self.x2_shared = x2_in
        self.y2_in = y2_in

        def local_objective_disc1(x1_local):
            # We want to minimize the global objective, so Disc1 tries to make y1 small
            # while respecting the linking variable y2_in
            current_y1 = x1_local[0]**2 + self.x2_shared + self.y2_in
            # Add a penalty if y1 is too large, or if x1 is out of bounds
            # For simplicity, let's just aim for a low y1
            return current_y1 # This is not the global objective, but Disc1's contribution

        # Bounds for x1
        bounds_x1 = [(0.0, 10.0)]

        # Initial guess for x1
        x1_init = np.array([self.x1])

        res = minimize(local_objective_disc1, x1_init, bounds=bounds_x1, method='Nelder-Mead')
        self.x1 = res.x[0]
        self.y1_out = self.evaluate(self.x2_shared, self.y2_in) # Recalculate y1 with new x1
        return self.y1_out, res.fun

class SellarDiscipline2: # Aerodynamics
    def __init__(self):
        self.x2 = 1.0 # Local design variable for Disc2 (could be shared, but optimized locally here)
        self.y1_in = 1.0 # Linking variable input from Disc1
        self.y2_out = 0.0 # Linking variable output to Disc1

    def evaluate(self, x2_in, y1_in):
        # Update inputs
        self.x2 = x2_in # x2 is now treated as a local design variable for Disc2
        self.y1_in = y1_in
        # Calculate output y2. Ensure y1_in is positive for sqrt.
        # This is where the spiky behavior can come from if y1_in goes negative.
        self.y2_out = np.sqrt(max(0.001, self.y1_in)) + self.x2
        return self.y2_out

    def optimize(self, x2_in, y1_in):
        # Disc2 optimizes its local variable x2.
        # It might try to minimize its contribution to the global objective,
        # or satisfy some local constraint.

        self.x2 = x2_in # Use the current x2 as the starting point for optimization
        self.y1_in = y1_in

        def local_objective_disc2(x2_local):
            # Disc2 tries to minimize its contribution to the global objective
            # For example, by making y2 small, but respecting y1_in
            current_y2 = np.sqrt(max(0.001, self.y1_in)) + x2_local[0]
            # Global objective has exp(-y2), so Disc2 wants y2 to be large.
            # This is a source of conflict!
            return -current_y2 # Try to maximize y2 (minimize -y2)

        # Bounds for x2
        bounds_x2 = [(0.0, 10.0)]

        # Initial guess for x2
        x2_init = np.array([self.x2])

        res = minimize(local_objective_disc2, x2_init, bounds=bounds_x2, method='Nelder-Mead')
        self.x2 = res.x[0]
        self.y2_out = self.evaluate(self.x2, self.y1_in) # Recalculate y2 with new x2
        return self.y2_out, res.fun

# --- MDO Coordinator for Sellar ---
class SellarMDOCoordinator:
    def __init__(self, disc1, disc2, max_iterations=150, tolerance=1e-4):
        self.disc1 = disc1
        self.disc2 = disc2
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.inconsistency_history = []
        self.y1_history = []
        self.y2_history = []
        self.x1_history = []
        self.x2_history = [] # For the shared design variable

    def coordinate(self):
        print("Starting Sellar MDO Coordination...")

        # Initial guesses for linking variables and shared design variable
        # These are the *inputs* that each discipline receives from the *other* or from the coordinator
        current_y1_for_disc2 = 100.0
        current_y2_for_disc1 = 50.0
        current_x2_shared = 1.0 # This x2 is the *input* to both disciplines

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")

            # Store previous values for inconsistency calculation
            prev_y1_from_disc1 = self.disc1.y1_out
            prev_y2_from_disc2 = self.disc2.y2_out

            # 1. Discipline 1 optimizes its local variable (x1) and calculates y1
            # It receives current_x2_shared and current_y2_for_disc1 as inputs
            new_y1_output, obj1_val = self.disc1.optimize(current_x2_shared, current_y2_for_disc1)
            self.x1_history.append(self.disc1.x1)
            print(f"Disc1 - x1: {self.disc1.x1:.4f}, y1_out: {new_y1_output:.4f}, Obj_D1: {obj1_val:.4f}")

            # Update the input for Discipline 2 for the next step
            current_y1_for_disc2 = new_y1_output

            # 2. Discipline 2 optimizes its local variable (x2) and calculates y2
            # It receives current_x2_shared (as its own x2) and current_y1_for_disc2 as inputs
            new_y2_output, obj2_val = self.disc2.optimize(current_x2_shared, current_y1_for_disc2)
            self.x2_history.append(self.disc2.x2) # Note: Disc2 updates its own x2
            print(f"Disc2 - x2: {self.disc2.x2:.4f}, y2_out: {new_y2_output:.4f}, Obj_D2: {obj2_val:.4f}")

            # Update the input for Discipline 1 for the next step
            current_y2_for_disc1 = new_y2_output

            # --- Inconsistency Calculation ---
            # We are looking for consistency in the linking variables y1 and y2.
            # y1_out from Disc1 should be equal to y1_in for Disc2 (which is current_y1_for_disc2)
            # y2_out from Disc2 should be equal to y2_in for Disc1 (which is current_y2_for_disc1)

            # The "true" y1 for this iteration is new_y1_output
            # The "true" y2 for this iteration is new_y2_output

            # Inconsistency is the difference between the *received* linking variable
            # and the *produced* linking variable by the discipline.
            # Or, more simply, how much the linking variables changed since the last iteration.

            # We need to compare the *output* of each discipline with the *input* it received
            # in the *previous* iteration.

            # Let's track the change in the outputs of the disciplines
            inconsistency_y1 = np.abs(new_y1_output - prev_y1_from_disc1) if i > 0 else 0.0
            inconsistency_y2 = np.abs(new_y2_output - prev_y2_from_disc2) if i > 0 else 0.0

            # Max inconsistency for this iteration
            current_max_inconsistency = max(inconsistency_y1, inconsistency_y2)
            self.inconsistency_history.append(current_max_inconsistency)

            # Store all values for plotting
            self.y1_history.append(new_y1_output)
            self.y2_history.append(new_y2_output)

            print(f"Max Inconsistency: {current_max_inconsistency:.6f}")

            if current_max_inconsistency < self.tolerance and i > 0: # i>0 to ensure at least one full cycle
                print(f"\nConvergence achieved in {i+1} iterations!")
                break
        else:
            print("\nMaximum iterations reached without convergence.")

        print("\n--- Final Results ---")
        print(f"Final x1: {self.disc1.x1:.4f}")
        print(f"Final x2: {self.disc2.x2:.4f}")
        print(f"Final y1 output: {self.disc1.y1_out:.4f}")
        print(f"Final y2 output: {self.disc2.y2_out:.4f}")

        # Calculate global objective and constraints at convergence
        final_f = self.disc1.x1**2 + self.disc1.y1_out + np.exp(-self.disc2.y2_out)
        final_g1 = 3.16 - self.disc1.y1_out
        final_g2 = self.disc2.y2_out - 24.0
        print(f"Final Global Objective (f): {final_f:.4f}")
        print(f"Final Constraint g1 (<=0): {final_g1:.4f}")
        print(f"Final Constraint g2 (<=0): {final_g2:.4f}")

        return self.disc1.x1, self.disc2.x2, self.disc1.y1_out, self.disc2.y2_out

# Instantiate Sellar disciplines
sellar_disc1 = SellarDiscipline1()
sellar_disc2 = SellarDiscipline2()

# Instantiate and run Sellar coordinator
sellar_coordinator = SellarMDOCoordinator(sellar_disc1, sellar_disc2, max_iterations=200, tolerance=1e-6)
final_x1, final_x2, final_y1, final_y2 = sellar_coordinator.coordinate()

# --- Plotting the Inconsistency History for Sellar ---
plt.figure(figsize=(12, 6))
plt.plot(sellar_coordinator.inconsistency_history, marker='o', linestyle='-', color='red')
plt.title('Sellar MDO Coordination: Maximum Inconsistency')
plt.xlabel('Iteration Number')
plt.ylabel('Max Inconsistency (Absolute Difference)')
# plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.axhline(y=sellar_coordinator.tolerance, color='blue', linestyle=':', label='Tolerance')
plt.legend()
plt.show()

# --- Plotting Linked Variable History for Sellar ---
plt.figure(figsize=(12, 6))
plt.plot(sellar_coordinator.y1_history, marker='x', linestyle='--', label='y1 (output from Disc1)')
plt.plot(sellar_coordinator.y2_history, marker='o', linestyle='-', label='y2 (output from Disc2)')
plt.title('Sellar MDO Coordination: Linked Variables')
plt.xlabel('Iteration Number')
plt.ylabel('Variable Value')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()