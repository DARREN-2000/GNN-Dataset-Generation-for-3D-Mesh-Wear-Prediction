import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

def create_dataset_structure(base_dir='wear_dataset', num_samples=30, stress_file=None):
    """
    Create a complete dataset structure for wear prediction
    
    Args:
        base_dir: Base directory for the dataset
        num_samples: Number of samples to generate
        stress_file: File containing stress values
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Load existing stress values if provided
    stress_values = None
    if stress_file and os.path.exists(stress_file):
        # Try multiple approaches to read the stress file
        try:
            # First try tab-separated format
            stress_df = pd.read_csv(stress_file, sep='\t')
            # Check if we have the expected column
            if 'MaxValue' in stress_df.columns:
                stress_values = stress_df['MaxValue'].values
                print(f"Loaded {len(stress_values)} stress values from {stress_file} (tab-separated)")
            else:
                # Try to find a column with stress values
                numeric_columns = stress_df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    value_col = numeric_columns[-1]  # Take the last numeric column
                    stress_values = stress_df[value_col].values
                    print(f"Loaded {len(stress_values)} stress values from column '{value_col}'")
        except Exception as e:
            # If that fails, try comma-separated
            try:
                stress_df = pd.read_csv(stress_file, sep=',')
                # Check columns again
                if 'MaxValue' in stress_df.columns:
                    stress_values = stress_df['MaxValue'].values
                    print(f"Loaded {len(stress_values)} stress values from {stress_file} (comma-separated)")
                else:
                    # Try to find a column with stress values
                    numeric_columns = stress_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        value_col = numeric_columns[-1]  # Take the last numeric column
                        stress_values = stress_df[value_col].values
                        print(f"Loaded {len(stress_values)} stress values from column '{value_col}'")
            except Exception as e2:
                # Last try - just load as raw data
                try:
                    raw_data = np.loadtxt(stress_file, delimiter=',', skiprows=1)
                    if raw_data.ndim > 1 and raw_data.shape[1] > 1:
                        stress_values = raw_data[:, -1]  # Last column
                    else:
                        stress_values = raw_data
                    print(f"Loaded {len(stress_values)} stress values as raw data")
                except Exception as e3:
                    print(f"Could not load stress file: {e3}")
                    print("Using synthetic stress values instead")
    
    # Generate samples
    for sample_idx in range(1, num_samples + 1):
        sample_dir = os.path.join(base_dir, f'sample_{sample_idx:02d}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Determine complexity of this sample's mesh
        # For first sample with stress values, use exactly the same number of nodes
        if sample_idx == 1 and stress_values is not None:
            # Calculate grid size to approximately match number of stress values
            grid_size = max(3, int(np.sqrt(len(stress_values))))
        else:
            # Vary mesh density across other samples
            grid_size = np.random.randint(8, 15)
        
        # Generate nodes on a grid with some noise
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Add some randomness to node positions (more for higher sample numbers)
        if sample_idx > 1:  # Keep first sample regular if we have stress values
            noise_level = 0.02 * (1 + 0.5 * (sample_idx / num_samples))
            xx += np.random.normal(0, noise_level, xx.shape)
            yy += np.random.normal(0, noise_level, yy.shape)
        
        # Z coordinates (mostly zero for 2D-like mesh, with some variations)
        zz = np.zeros_like(xx)
        
        # Some samples will have slight 3D curvature
        if sample_idx % 5 == 0:  # Every 5th sample has some curvature
            zz = 0.05 * np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
        
        # Combine coordinates into nodes array
        nodes = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        # If using stress values for first sample, make sure node count matches exactly
        if sample_idx == 1 and stress_values is not None:
            if len(nodes) > len(stress_values):
                # Truncate nodes to match stress values
                nodes = nodes[:len(stress_values)]
            elif len(nodes) < len(stress_values):
                # Add additional nodes to match stress values
                extra_nodes = len(stress_values) - len(nodes)
                extra_x = np.random.uniform(0, 1, extra_nodes)
                extra_y = np.random.uniform(0, 1, extra_nodes)
                extra_z = np.zeros(extra_nodes)
                extra_points = np.column_stack([extra_x, extra_y, extra_z])
                nodes = np.vstack([nodes, extra_points])
        
        # Save nodes to file
        np.savetxt(os.path.join(sample_dir, 'nodes.csv'), nodes, delimiter=',')
        
        # Generate elements (quadrilateral elements)
        if sample_idx == 1 and stress_values is not None and grid_size * grid_size != len(nodes):
            # For the first sample with stress values, create a mesh that works with the node count
            elements = []
            remaining_nodes = len(nodes)
            
            # We'll create a triangular mesh instead for arbitrary node counts
            from scipy.spatial import Delaunay
            tri = Delaunay(nodes[:, :2])
            elements = tri.simplices
        else:
            # Regular grid elements
            elements = []
            for i in range(grid_size - 1):
                for j in range(grid_size - 1):
                    # Get indices of the four corners of each quad element
                    n1 = i * grid_size + j
                    n2 = i * grid_size + (j + 1)
                    n3 = (i + 1) * grid_size + (j + 1)
                    n4 = (i + 1) * grid_size + j
                    elements.append([n1, n2, n3, n4])
        
        # Save elements to file
        np.savetxt(os.path.join(sample_dir, 'elements.csv'), elements, delimiter=',', fmt='%d')
        
        # Generate material properties with some variation between samples
        # Base properties with random variations
        thickness_base = 0.05  # meters
        youngs_modulus_base = 2.1e11  # Pa (steel)
        density_base = 7800  # kg/m³ (steel)
        
        # Add variations for different samples
        thickness = thickness_base * np.random.uniform(0.8, 1.2)
        youngs_modulus = youngs_modulus_base * np.random.uniform(0.9, 1.1)
        density = density_base * np.random.uniform(0.95, 1.05)
        
        # Number of roller paths (1-3)
        num_roller_paths = np.random.randint(1, 4)
        
        # Save properties to file
        properties = np.array([thickness, youngs_modulus, density, num_roller_paths])
        np.savetxt(os.path.join(sample_dir, 'properties.csv'), properties, delimiter=',')
        
        # Generate roller paths - FIX for array dimension mismatch
        for path_idx in range(num_roller_paths):
            # Create paths with different patterns
            num_points = 20  # Consistent number of points
            t = np.linspace(0, 1, num_points)
            
            if path_idx == 0:
                # Straight path
                path_x = np.ones(num_points) * (0.2 + 0.6 * path_idx / max(1, num_roller_paths - 1))
                path_y = t
                path_z = np.zeros(num_points)
            elif path_idx == 1:
                # Diagonal path
                path_x = t
                path_y = t
                path_z = np.zeros(num_points)
            else:
                # Curved path
                path_x = 0.5 + 0.3 * np.sin(2 * np.pi * t)
                path_y = t
                path_z = np.zeros(num_points)
            
            # Verify all arrays have the same length before stacking
            assert len(path_x) == len(path_y) == len(path_z), "Path arrays must have same length"
            
            # Combine into path
            path = np.column_stack([path_x, path_y, path_z])
            
            # Save path
            np.savetxt(os.path.join(sample_dir, f'roller_path_{path_idx+1}.csv'), path, delimiter=',')
        
        # Calculate wear values
        # For first sample, use the provided stress values
        if sample_idx == 1 and stress_values is not None and len(stress_values) == len(nodes):
            # Convert stress values directly to wear
            # Normalize stress values to a reasonable wear range
            normalized_stress = (stress_values - np.min(stress_values)) / (np.max(stress_values) - np.min(stress_values) + 1e-10)
            wear = normalized_stress * 0.001  # Scale to mm range which is typical for wear
            
        else:
            # Generate wear values based on physics:
            # Calculate distances to roller paths
            min_distances = np.ones(len(nodes)) * np.inf
            for path_idx in range(num_roller_paths):
                path_file = os.path.join(sample_dir, f'roller_path_{path_idx+1}.csv')
                roller_path = np.loadtxt(path_file, delimiter=',')
                
                for i, node in enumerate(nodes):
                    for point in roller_path:
                        dist = np.linalg.norm(node[:2] - point[:2])  # Distance in xy plane
                        min_distances[i] = min(min_distances[i], dist)
            
            # Base wear calculation
            wear = 0.001 * np.exp(-10 * min_distances)  # Exponential decay with distance
            
            # Adjust for material properties
            wear *= (0.05 / thickness)  # Thinner material = more wear
            wear *= (2.1e11 / youngs_modulus)  # Lower Young's modulus = more wear
            
            # Add some randomness
            wear *= np.random.normal(1.0, 0.2, len(wear))
        
        # Ensure wear is always positive
        wear = np.clip(wear, 0, None)
        
        # Save wear values
        np.savetxt(os.path.join(sample_dir, 'wear.csv'), wear, delimiter=',')
        
        # For visualization purposes, save a plot of the mesh with wear values
        plt.figure(figsize=(10, 8))
        
        # Plot mesh edges (simplified for visualization)
        for element in elements:
            for i in range(len(element)):
                j = (i + 1) % len(element)
                plt.plot([nodes[element[i], 0], nodes[element[j], 0]], 
                         [nodes[element[i], 1], nodes[element[j], 1]], 
                         'k-', linewidth=0.3)
        
        # Plot wear values
        scatter = plt.scatter(nodes[:, 0], nodes[:, 1], c=wear, cmap='hot', s=50, alpha=0.8)
        
        # Plot roller paths
        for path_idx in range(num_roller_paths):
            path_file = os.path.join(sample_dir, f'roller_path_{path_idx+1}.csv')
            roller_path = np.loadtxt(path_file, delimiter=',')
            plt.plot(roller_path[:, 0], roller_path[:, 1], 'b-', linewidth=2, label=f'Roller Path {path_idx+1}')
        
        plt.colorbar(scatter, label='Wear')
        plt.title(f'Sample {sample_idx} - Mesh and Wear Values')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'visualization.png'))
        plt.close()
            
    print(f"Successfully generated {num_samples} samples in {base_dir}")
    print(f"Each sample includes nodes, elements, material properties, roller paths, and wear values")
    print(f"The dataset is ready to use with the wear prediction GNN model")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dataset for wear prediction GNN')
    parser.add_argument('--output_dir', type=str, default='wear_dataset',
                       help='Directory to save the generated dataset')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to generate')
    parser.add_argument('--stress_file', type=str, default=None,
                       help='File containing stress values')
    
    args = parser.parse_args()
    
    create_dataset_structure(args.output_dir, args.num_samples, args.stress_file)