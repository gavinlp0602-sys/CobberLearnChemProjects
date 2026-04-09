import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import fetch_california_housing # Note: fixed 'cal_housing' to 'california_housing'
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & MODEL TRAINING ---
# Ensure the function name is correct: fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names  # Fixed the typo here

# Calculate means for the defaults
feature_defaults = np.mean(X, axis=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)


# --- 2. GUI CLASS ---
class HousingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("California House Price Predictor")
        self.root.geometry("450x650")
        self.root.configure(padx=20, pady=20)

        tk.Label(root, text="House Feature Inputs", font=("Helvetica", 14, "bold")).pack(pady=10)

        # Create input fields with defaults
        self.entries = {}
        for i, feature in enumerate(feature_names):
            frame = tk.Frame(root)
            frame.pack(fill="x", pady=2)

            lbl = tk.Label(frame, text=f"{feature}:", width=15, anchor="w")
            lbl.pack(side="left")

            entry = tk.Entry(frame)
            # Insert the mean value as a default, rounded for readability
            entry.insert(0, f"{feature_defaults[i]:.2f}")
            entry.pack(side="right", expand=True, fill="x")
            self.entries[feature] = entry

        # --- BUTTON SECTION ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=20)

        # Predict Button
        self.predict_btn = tk.Button(btn_frame, text="Predict Price", command=self.get_prediction,
                                     bg="#4CAF50", fg="white", width=15, font=("Helvetica", 10, "bold"))
        self.predict_btn.grid(row=0, column=0, padx=5)

        # Clear Button
        self.clear_btn = tk.Button(btn_frame, text="Clear All", command=self.clear_fields,
                                   bg="#f44336", fg="white", width=15)
        self.clear_btn.grid(row=0, column=1, padx=5)

        # Result Display
        self.result_label = tk.Label(root, text="Predicted Value: ---", font=("Helvetica", 13, "italic"))
        self.result_label.pack(pady=20)

    def validate_inputs(self):
        """Checks if all inputs are valid floats."""
        try:
            input_values = []
            for feature in feature_names:
                raw_val = self.entries[feature].get().strip()
                if not raw_val:
                    raise ValueError(f"{feature} is empty.")
                input_values.append(float(raw_val))
            return input_values
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {e}\nPlease enter numbers only.")
            return None

    def get_prediction(self):
        # Trigger Validation
        validated_data = self.validate_inputs()

        if validated_data:
            input_array = np.array([validated_data])
            input_scaled = scaler.transform(input_array)

            prediction = rf_model.predict(input_scaled)[0]
            # Convert $100k blocks to actual currency
            final_price = prediction * 100000

            self.result_label.config(text=f"Predicted Value: ${final_price:,.2f}",
                                     fg="#2E7D32", font=("Helvetica", 13, "bold"))

    def clear_fields(self):
        """Clears all text boxes and resets the result label."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="Predicted Value: ---", fg="black", font=("Helvetica", 13, "italic"))


# --- 3. RUN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = HousingApp(root)
    root.mainloop()