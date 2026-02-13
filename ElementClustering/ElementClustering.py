import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)


def get_data(choice):
    # Data for Group 1
    alkali = {
        "Name": ["Lithium", "Sodium", "Potassium", "Rubidium", "Cesium", "Francium"],
        "Symbol": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
        "Group": "Alkali Metal",
        "Atomic Radius (pm)": [152, 186, 227, 248, 265, 348],
        "First Ionization Energy (kJ/mol)": [520.2, 495.8, 418.8, 403.0, 375.7, 393.0]
    }
    # Data for Group 17
    halogens = {
        "Name": ["Fluorine", "Chlorine", "Bromine", "Iodine", "Astatine"],
        "Symbol": ["F", "Cl", "Br", "I", "At"],
        "Group": "Halogen",
        "Atomic Radius (pm)": [42, 79, 94, 115, 127],
        "First Ionization Energy (kJ/mol)": [1681.0, 1251.2, 1139.9, 1008.4, 890.0]
    }

    if choice == '1':
        return pd.DataFrame(alkali)
    elif choice == '2':
        return pd.DataFrame(halogens)
    else:
        # Combine both if choice is '3'
        return pd.concat([pd.DataFrame(alkali), pd.DataFrame(halogens)], ignore_index=True)


def main():
    print("--- Periodic Table Clustering Tool ---")

    while True:
        print("\nSelect Dataset:")
        print("1: Alkali Metals (Group 1)")
        print("2: Halogens (Group 17)")
        print("3: Both Groups Combined")
        print("Type 'exit' to quit.")

        choice = input("\nYour choice (1/2/3): ").strip().lower()

        if choice in ['exit', 'stop', 'quit']:
            print("Exiting...")
            sys.exit()

        if choice not in ['1', '2', '3']:
            print("Invalid selection. Please try again.")
            continue

        df = get_data(choice)

        try:
            k_input = input(f"Enter number of clusters (k) [Max {len(df)}]: ").strip()
            k = int(k_input)

            if k < 1 or k > len(df):
                print(f"Please enter a number between 1 and {len(df)}.")
                continue

            # Clustering logic
            X = df[["Atomic Radius (pm)", "First Ionization Energy (kJ/mol)"]]
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df['Cluster'] = kmeans.fit_predict(X)

            # Plotting
            plt.figure(figsize=(9, 6))
            plt.scatter(df["Atomic Radius (pm)"], df["First Ionization Energy (kJ/mol)"],
                        c=df['Cluster'], cmap='plasma', s=100, edgecolors='black')

            for i, txt in enumerate(df["Symbol"]):
                plt.annotate(txt, (df["Atomic Radius (pm)"][i], df["First Ionization Energy (kJ/mol)"][i]),
                             xytext=(0, 8), textcoords='offset points', ha='center')

            plt.title(f"Clustering Result (k={k})")
            plt.xlabel("Atomic Radius (pm)")
            plt.ylabel("Ionization Energy (kJ/mol)")
            plt.grid(True, alpha=0.3)
            plt.show()

            print("\n--- Current Data View ---")
            print(df[['Name', 'Group', 'Cluster']])

        except ValueError:
            print("Error: Please enter a valid integer for k.")


if __name__ == "__main__":
    main()