# GNN Dataset Generation for 3D Mesh Wear Prediction

## Overview

This document describes how to generate synthetic 3D mesh datasets for training and evaluating Graph Neural Networks (GNNs) for wear prediction. The dataset generator creates realistic 3D mesh structures with customizable properties, simulated wear patterns, and prepares them in the format required by the GNN model.

## Dataset Generation Features

- Generate parametrically defined 3D mesh geometries
- Customize mesh density and complexity
- Apply realistic material properties (thickness, Young's modulus)
- Simulate various wear patterns based on physical principles
- Support for multiple sample generation with controlled variation

## Installation Requirements

```bash
pip install numpy==1.24.4
pip install pandas==1.5.3
pip install matplotlib==3.5.2
pip install scipy==1.9.1
```

## Basic Usage

To generate a standard dataset with 30 samples:

```bash
python generate_wear_dataset.py --num-samples 30 --output-dir wear_dataset
```

## Advanced Usage

### Customizing Mesh Parameters

```bash
python generate_wear_dataset.py \
    --num-samples 20 \
    --min-nodes 64 \
    --max-nodes 196 \
    --output-dir custom_mesh_dataset \
    --add-noise 0.05
```

### Customizing Material Properties

```bash
python generate_wear_dataset.py \
    --num-samples 15 \
    --thickness-range 0.5 2.0 \
    --youngs-range 190 210 \
    --density-range 7.7 7.9 \
    --output-dir custom_material_dataset
```

### Customizing Wear Patterns

```bash
python generate_wear_dataset.py \
    --num-samples 10 \
    --wear-pattern "gaussian" \
    --wear-intensity 1.5 \
    --output-dir gaussian_wear_dataset
```

## Generated Dataset Structure

The generated dataset follows this structure:

```
output_dir/
├── sample_01/
│   ├── nodes.csv       # Node coordinates (x,y,z)
│   ├── elements.csv    # Element connectivity
│   ├── properties.csv  # Material properties
│   └── wear.csv        # Target wear values for each node
├── sample_02/
│   ├── ...
...
```

### File Formats

#### nodes.csv
```
node_id,x,y,z
1,0.0,0.0,0.0
2,1.0,0.0,0.0
...
```

#### elements.csv
```
element_id,node1,node2,node3,node4
1,1,2,11,10
2,2,3,12,11
...
```

#### properties.csv
```
property,value
thickness,1.2
youngs_modulus,205.0
density,7.85
roller_paths,3
```

#### wear.csv
```
node_id,wear_value
1,0.00012
2,0.00024
...
```

## Available Mesh Types

The generator supports several primitive mesh types:

1. **Rectangular Plate**: Basic flat plate with variable dimensions
2. **Curved Surface**: Surface with controlled curvature
3. **Cylindrical Section**: Section of a cylinder with variable radius and angle
4. **Spherical Cap**: Section of a sphere with variable radius
5. **Custom Mesh**: Import your own mesh from STL files

Example for generating a specific mesh type:

```bash
python generate_wear_dataset.py --mesh-type "curved_surface" --curvature 0.2
```

## Available Wear Patterns

The generator includes several wear pattern models:

1. **Uniform**: Constant wear across the surface
2. **Linear**: Wear increases linearly along one axis
3. **Gaussian**: Wear concentrated around specified points
4. **Distance-Based**: Wear based on distance from specified paths
5. **Material-Based**: Wear dependent on material properties
6. **Combined**: Combination of multiple wear patterns

Example:

```bash
python generate_wear_dataset.py --wear-pattern "combined" --pattern-weights 0.3 0.7
```

## Customizing the Generation Process

### Adding Custom Noise and Deformation

For more realistic data, add noise and deformation:

```bash
python generate_wear_dataset.py --geometric-noise 0.02 --wear-noise 0.01
```

### Controlling Mesh Resolution

```bash
python generate_wear_dataset.py --mesh-resolution "high" --output-dir high_res_dataset
```

Or specify exact divisions:

```bash
python generate_wear_dataset.py --x-divisions 15 --y-divisions 15
```

## Batch Generation for Parameter Sweeps

To perform parameter studies, use the batch generation script:

```bash
python batch_generate_datasets.py --parameter thickness --range 0.5 2.0 0.5
```

This will generate multiple datasets with thickness values: 0.5, 1.0, 1.5, 2.0.

## Visualizing Generated Samples

```bash
python visualize_samples.py --dataset-dir wear_dataset --sample-ids 1 5 10
```

This will generate plots showing the mesh geometry and wear patterns for samples 1, 5, and 10.

## Integrating with the GNN Model

After generating the dataset, train the GNN model:

```bash
python GNN.py --custom-dataset --dataset-dir wear_dataset --batch-size 1
```

## Common Issues and Solutions

1. **Memory Errors**: Reduce the mesh resolution or number of samples
   ```bash
   python generate_wear_dataset.py --mesh-resolution "low" --num-samples 10
   ```

2. **Unrealistic Wear Patterns**: Adjust wear intensity and distribution
   ```bash
   python generate_wear_dataset.py --wear-intensity 0.8 --wear-distribution "exponential"
   ```

3. **Mesh Quality Issues**: Enable mesh quality improvements
   ```bash
   python generate_wear_dataset.py --improve-mesh-quality
   ```

## Example Workflow

```bash
# 1. Generate a dataset with 30 samples
python generate_wear_dataset.py --num-samples 30 --output-dir wear_dataset

# 2. Visualize sample 5 to verify generation
python visualize_samples.py --dataset-dir wear_dataset --sample-ids 5

# 3. Train the GNN model on the generated dataset
python GNN.py --custom-dataset --dataset-dir wear_dataset
```

## Citation

If you use this dataset generator in your research, please cite:

## License

This dataset generation tool is licensed under the MIT License - see the LICENSE file for details.
