import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_camera_parameters(focal_length=500, image_width=640, image_height=480):
    """Create camera intrinsic and extrinsic parameters."""
    intrinsic_matrix = np.array([
        [focal_length, 0, image_width/2],
        [0, focal_length, image_height/2],
        [0, 0, 1]
    ])
    
    rvec = np.array([0, 0, 0], dtype=np.float32)
    tvec = np.array([0, 0, 100], dtype=np.float32)
    
    return intrinsic_matrix, rvec, tvec

def generate_paraboloid_points(u_range=(-1, 1), v_range=(-1, 1), num_points=20):
    """Generate 3D points on a paraboloid surface."""
    u = np.linspace(u_range[0], u_range[1], num=num_points)
    v = np.linspace(v_range[0], v_range[1], num=num_points)
    u, v = np.meshgrid(u, v)
    
    x = u
    y = v
    z = u**2 + v**2
    
    points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return x, y, z, points_3d

def project_3d_to_2d(points_3d, rvec, tvec, intrinsic_matrix):
    """Project 3D points onto 2D image plane."""
    points_2d, _ = cv2.projectPoints(points_3d,
                                     rvec, tvec.reshape(-1, 1),
                                     intrinsic_matrix, None)
    return points_2d

def create_2d_projection_image(points_2d, image_height, image_width):
    """Create 2D image with projected points."""
    img = np.zeros((image_height, image_width), dtype=np.uint8)
    for point in points_2d.astype(int):
        img = cv2.circle(img, tuple(point[0]), 2, 255, -1)
    return img
def plot_visualization(points_3d, x=None, y=None, z=None, img=None, plot_type='3d', figsize=(8, 6)):
    """Create visualization figure with standardized aesthetics."""
    # Standard configuration
    config = {
        'title_size': 14,
        'fontsize': 12,
        'scatter_size': 25,  # Average of previous sizes (20-30)
        'scatter_alpha': 0.65,  # Average of previous alphas (0.6-0.7)
        'colormap': 'viridis',
        'grid_alpha': 0.3
    }
    
    fig = plt.figure(figsize=figsize)
    
    if plot_type == '3d' and x is not None and y is not None and z is not None:
        # 3D Surface plot
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap=config['colormap'], alpha=0.8)
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c='red', s=config['scatter_size'], alpha=config['scatter_alpha'])
        
        ax.set_xlabel('X', fontsize=config['fontsize'])
        ax.set_ylabel('Y', fontsize=config['fontsize']) 
        ax.set_zlabel('Z', fontsize=config['fontsize'])
        ax.set_title('3D Paraboloid Surface', fontsize=config['title_size'])
        ax.view_init(elev=20, azim=45)
        
    elif plot_type == '2d_projection' and img is not None:
        # 2D Projection plot
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        ax.set_title('2D Projection', fontsize=config['title_size'])
        ax.set_xlabel('X (pixels)', fontsize=config['fontsize'])
        ax.set_ylabel('Y (pixels)', fontsize=config['fontsize'])
        ax.axis('on')
        
    elif plot_type in ['xy', 'yz', 'zx']:
        # 2D Scatter plots
        ax = fig.add_subplot(111)
        
        # Set indices and labels based on projection type
        if plot_type == 'xy':
            x_idx, y_idx, c_idx = 0, 1, 2
            xlabel, ylabel = 'X', 'Y'
            title = 'XY Projection (Top View)'
            cbar_label = 'Z height'
        elif plot_type == 'yz':
            x_idx, y_idx, c_idx = 1, 2, 0
            xlabel, ylabel = 'Y', 'Z'
            title = 'YZ Projection (Side View)'
            cbar_label = 'X position'
        else:  # zx
            x_idx, y_idx, c_idx = 2, 0, 1
            xlabel, ylabel = 'Z', 'X'
            title = 'ZX Projection (Front View)'
            cbar_label = 'Y position'
            
        scatter = ax.scatter(points_3d[:, x_idx], points_3d[:, y_idx],
                           c=points_3d[:, c_idx], cmap=config['colormap'],
                           s=config['scatter_size'], alpha=config['scatter_alpha'])
        
        ax.set_xlabel(xlabel, fontsize=config['fontsize'])
        ax.set_ylabel(ylabel, fontsize=config['fontsize'])
        ax.set_title(title, fontsize=config['title_size'])
        ax.grid(True, alpha=config['grid_alpha'])
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label=cbar_label)
    
    plt.tight_layout()
    plt.show(block=False)
    return fig

def create_all_visualizations(points_3d, x, y, z, img):
    """Create all visualization figures using standardized configuration."""
    figures = []
    
    # Create each type of visualization
    figures.append(plot_visualization(points_3d, x, y, z, plot_type='3d'))
    figures.append(plot_visualization(points_3d, img=img, plot_type='2d_projection'))
    figures.append(plot_visualization(points_3d, plot_type='xy'))
    figures.append(plot_visualization(points_3d, plot_type='yz'))
    figures.append(plot_visualization(points_3d, plot_type='zx'))
    
    return figures

def main():
    """Main function with simplified configuration."""
    
    # Global configuration
    config = {
        'figsize': (8, 6),
        '3d': {
            'title': '3D Paraboloid Surface',
            'xlabel': 'X', 'ylabel': 'Y', 'zlabel': 'Z',
            'colormap': 'viridis', 'alpha': 0.8,
            'scatter_color': 'red', 'scatter_size': 20, 'scatter_alpha': 0.6,
            'elev': 20, 'azim': 45,
            'fontsize': 12, 'title_size': 14
        },
        '2d': {
            'fontsize': 12, 'title_size': 14
        },
        'scatter_plots': [
            {
                'title': 'XY Projection (Top View)',
                'xlabel': 'X', 'ylabel': 'Y',
                'colormap': 'viridis', 'scatter_size': 30, 'scatter_alpha': 0.7,
                'grid_alpha': 0.3, 'colorbar_label': 'Z height',
                'x_idx': 0, 'y_idx': 1, 'color_idx': 2,
                'fontsize': 12, 'title_size': 14
            },
            {
                'title': 'YZ Projection (Side View)',
                'xlabel': 'Y', 'ylabel': 'Z',
                'colormap': 'plasma', 'scatter_size': 30, 'scatter_alpha': 0.7,
                'grid_alpha': 0.3, 'colorbar_label': 'X position',
                'x_idx': 1, 'y_idx': 2, 'color_idx': 0,
                'fontsize': 12, 'title_size': 14
            },
            {
                'title': 'ZX Projection (Front View)',
                'xlabel': 'Z', 'ylabel': 'X',
                'colormap': 'coolwarm', 'scatter_size': 30, 'scatter_alpha': 0.7,
                'grid_alpha': 0.3, 'colorbar_label': 'Y position',
                'x_idx': 2, 'y_idx': 0, 'color_idx': 1,
                'fontsize': 12, 'title_size': 14
            }
        ]
    }
    
    print("Creating 3D to 2D mapping visualization...")
    
    # Create camera parameters
    intrinsic_matrix, rvec, tvec = create_camera_parameters(500, 640, 480)
    
    # Generate 3D points
    x, y, z, points_3d = generate_paraboloid_points((-1, 1), (-1, 1), 20)
    
    # Project 3D to 2D
    points_2d = project_3d_to_2d(points_3d, rvec, tvec, intrinsic_matrix)
    
    # Create 2D image
    img = create_2d_projection_image(points_2d, 480, 640)
    
    # Create all visualizations
    figures = create_all_visualizations(points_3d, x, y, z, img)
    
    # Show all figures
    plt.show()
    plt.ion()
    
    print(f"Created {len(figures)} visualization windows")

if __name__ == "__main__":
    main()