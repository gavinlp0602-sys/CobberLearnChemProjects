import matplotlib.pyplot as plt

# 1. Data for the first 10 linear alkanes
# Alkanes: Methane, Ethane, Propane, Butane, Pentane, Hexane, Heptane, Octane, Nonane, Decane
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
boiling_points_celsius = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.7, 150.8, 174.1]

# 2. Create the scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(carbons, boiling_points_celsius, color='blue', marker='o')

# 3. Add titles and labels
plt.title('Boiling Point of Linear Alkanes vs. Number of Carbons')
plt.xlabel('Number of Carbon Atoms')
plt.ylabel('Boiling Point (°C)')

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

#Save plot to directory
plt.savefig('alkane_boiling_points.png', dpi=300, bbox_inches='tight')

print("Plot saved successfully as 'alkane_boiling_points.png'")

# Display the plot
plt.show()
