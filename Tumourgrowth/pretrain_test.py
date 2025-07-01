import os
import torch
from data.Trans_data import create_dataloaders
from model.Pretrainmodel import Model
from utils.loss import setup_seed, count_params_with_grad,LpLoss
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


test_path = ".\\domain_test.npy"

test_loader = create_dataloaders(test_path, batch_size=1,shuffle = False, max_samples=600,num_random_points=300)




# 示例用法
myloss = LpLoss(size_average=False)
MSE_loss = torch.nn.MSELoss()
l2_loss = LpLoss(size_average=False)
l2_loss_train = LpLoss(size_average=False)

def f_loss(pred_contour, target_fd,num_freq=10):

    begin = target_fd[:, :, :num_freq]
    end = target_fd[:, :, -num_freq:]
    fft_coeffs = torch.cat((begin, end), dim=2)

    # num_freq = pred_fd.shape[2]
    complex_pred = pred_contour[:, :, :, 0] + 1j * pred_contour[:, :, :, 1]
    pred_fd = torch.fft.fft(complex_pred, dim=-1)  # No need to normalize, MSE handles scale.
    pred_begin = pred_fd[:, :, :num_freq]
    pred_end = pred_fd[:, :, -num_freq:]
    pred_coeffs = torch.cat((pred_begin, pred_end), dim=2)

    low_freq_loss_real = l2_loss(pred_coeffs.real, fft_coeffs.real)
    low_freq_loss_imag = l2_loss(pred_coeffs.imag, fft_coeffs.imag)

    return low_freq_loss_real,low_freq_loss_imag

def fourier_loss(pred_contour, target_fd, num_freq=64,name=  None):
    begin = target_fd[:, :, :num_freq]
    end = target_fd[:, :, -num_freq:]
    fft_coeffs = torch.cat((begin, end), dim=2)

    complex_pred = pred_contour[:, :,:, 0] + 1j * pred_contour[:, :,:, 1]
    pred_fd = torch.fft.fft(complex_pred,dim = -1)  # No need to normalize, MSE handles scale.
    pred_begin = pred_fd[:, :, :num_freq]
    pred_end = pred_fd[:, :, -num_freq:]
    pred_coeffs = torch.cat((pred_begin, pred_end), dim=2)

    if name is None:
        low_freq_loss_real = myloss(pred_coeffs.real,fft_coeffs.real)
        low_freq_loss_imag = myloss(pred_coeffs.imag, fft_coeffs.imag)
    else:
        low_freq_loss_real = l2_loss(pred_coeffs.real, fft_coeffs.real)
        low_freq_loss_imag = l2_loss(pred_coeffs.imag, fft_coeffs.imag)
    low_freq_loss = low_freq_loss_real + low_freq_loss_imag
    total_loss = low_freq_loss

    return total_loss
def plot_3d_interpolation(original_sections, original_values, num_new_layers=50,
                          original_z=None, num_longitudinal_lines=36,
                          elev=25, azim=45, save_path=None, dpi=600,batch_idx=0,name = "target"):


    # Set up original z coordinates if not provided
    if original_z is None:
        original_z = np.linspace(0, 100, original_sections.shape[0])
    if  batch_idx == 0 and name is not 'loss':
        original_sections[0] = np.roll(original_sections[0], -150, 0)

    # Create target z coordinates for interpolation
    target_z = np.linspace(original_z.min(), original_z.max(), num_new_layers)

    # Initialize arrays for interpolated data
    interpolated_sections = np.zeros((num_new_layers, original_sections.shape[1], 2))
    interpolated_values = np.zeros((num_new_layers, original_sections.shape[1]))

    # Perform interpolation for each point
    for point_idx in range(original_sections.shape[1]):
        x_vals = original_sections[:, point_idx, 0]
        y_vals = original_sections[:, point_idx, 1]
        value_vals = original_values[:, point_idx]

        interp_x = interp1d(original_z, x_vals, kind='cubic', fill_value="extrapolate")
        interp_y = interp1d(original_z, y_vals, kind='cubic', fill_value="extrapolate")
        interp_v = interp1d(original_z, value_vals, kind='cubic', fill_value="extrapolate")

        for layer_idx, z_val in enumerate(target_z):
            interpolated_sections[layer_idx, point_idx, 0] = interp_x(z_val)
            interpolated_sections[layer_idx, point_idx, 1] = interp_y(z_val)
            interpolated_values[layer_idx, point_idx] = interp_v(z_val)

    # Prepare all points for 3D plotting
    all_points = []
    for layer_idx in range(num_new_layers):
        for point_idx in range(original_sections.shape[1]):
            x = interpolated_sections[layer_idx, point_idx, 0]
            y = interpolated_sections[layer_idx, point_idx, 1]
            z = target_z[layer_idx]
            all_points.append([x, y, z])
    all_points = np.array(all_points)

    # Create triangles for surface
    triangles = []
    for layer_idx in range(num_new_layers - 1):
        for point_idx in range(original_sections.shape[1]):
            next_point_idx = (point_idx + 1) % original_sections.shape[1]
            idx1 = layer_idx * original_sections.shape[1] + point_idx
            idx2 = layer_idx * original_sections.shape[1] + next_point_idx
            idx3 = (layer_idx + 1) * original_sections.shape[1] + point_idx
            idx4 = (layer_idx + 1) * original_sections.shape[1] + next_point_idx

            triangles.append([idx1, idx2, idx3])
            triangles.append([idx3, idx2, idx4])

    # Calculate average values for each triangle
    triangle_values = []
    for tri in triangles:
        v1, v2, v3 = interpolated_values.flatten()[tri[0]], interpolated_values.flatten()[tri[1]], \
        interpolated_values.flatten()[tri[2]]
        triangle_values.append((v1 + v2 + v3) / 3)
    triangle_values = np.array(triangle_values)

    # Create figure
    fig = plt.figure(figsize=(2.73, 2.73), facecolor='none')
    ax = fig.add_subplot(111, projection='3d')

    # Make the axes background transparent
    ax.patch.set_alpha(0.0)

    # Plot surface
    surf = ax.plot_trisurf(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                           triangles=triangles, alpha=0.7,
                           linewidth=0.05, antialiased=True)
    surf.set_array(triangle_values)
    surf.set_cmap('YlGnBu')

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=20, format='%.3f', pad=-0.1)
    # cbar.set_label('Physical Value')
    cbar.ax.tick_params(labelsize=8)  # Adjust the size as needed
    vmin, vmax = surf.norm.vmin, surf.norm.vmax
    ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(ticks)
    # Add longitudinal lines
    for point_idx in range(0, original_sections.shape[1], original_sections.shape[1] // num_longitudinal_lines):
        xs = interpolated_sections[:, point_idx, 0]
        ys = interpolated_sections[:, point_idx, 1]
        zs = target_z
        ax.plot(xs, ys, zs, 'k-', linewidth=0.05, alpha=0.4)

    # Add original sections
    for i, section in enumerate(original_sections):
        z = original_z[i]
        x = section[:, 0]
        y = section[:, 1]
        ax.plot(x, y, z, linewidth=0.05, color='black')

    # Add base section
    vertices = [list(zip(
        original_sections[0, :, 0],
        original_sections[0, :, 1],
        np.full_like(original_sections[0, :, 0], original_z[0])
    ))]
    poly = Poly3DCollection(vertices, alpha=0.5, color='red')
    ax.add_collection3d(poly)

    # Configure plot
    ax.set_axis_off()
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    plt.tight_layout()

    # Save or show
    plt.savefig(f".\\3d_images\\{name}{batch_idx}.png",dpi=200,  bbox_inches='tight')
    # plt.show()

def generate_interior_points(batch_size,t, device):
    r = torch.sqrt(torch.rand(15, batch_size, device=device))  # [10, batch_size]
    theta = 2 * torch.pi * torch.rand(15, batch_size, device=device)  # [10, batch_size]
    x = r * torch.cos(theta)  # [10, batch_size]
    y = r * torch.sin(theta)  # [10, batch_size]
    t = t.squeeze().unsqueeze(1).expand(-1,  batch_size)
    points = torch.stack((x, y,t), dim=-1)  # [10, batch_size, 2]
    points = points.reshape(points.shape[0]*points.shape[1], -1)
    points = points.unsqueeze(0)
    return points
def vis_decoder_dis(embed_dim=64, seed=420, load_path=None, size=(10, 4)):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, num_points=1000)
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}. Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l2_errors = []
    abs_errors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):

            fx_test = batch.fx.to(device)
            init_domain_test = batch.initial_domain.to(device)
            t = batch.step.to(device)
            input_interior = generate_interior_points(2000, t[:1], device)
            x_interior, y_interior, t_interior = input_interior[:, :, 0], input_interior[:, :, 1], input_interior[:, :,2]
            x_pred_interior, y_pred_interior, x_de, y_de = m.forward(x=x_interior, y=y_interior, t=t_interior,
                                                                     fx=fx_test[:1], init_domain=init_domain_test[:1])

            decoder_loss = l2_loss(x_de,x_interior)
            decoder_loss2 = l2_loss(y_de,y_interior)
            batch_l2_errors = (decoder_loss + decoder_loss2)/2
            batch_abs_errors = (MSE_loss(x_de,x_interior)+MSE_loss(y_de,y_interior))/2

            l2_errors.extend(batch_l2_errors.unsqueeze(0).cpu().numpy())
            abs_errors.extend(batch_abs_errors.unsqueeze(0).cpu().numpy())
        l2_errors = np.array(l2_errors)
        abs_errors = np.array(abs_errors)

        # Calculate mean errors
        mean_l2 = np.mean(l2_errors)
        mean_abs = np.mean(abs_errors)
        print(f"Mean L2 error: {mean_l2}")
        print(f"Mean Absolute error: {mean_abs}")
        # Plot error distributions using KDE
        plt.figure(figsize=size, facecolor='none')  # Taller figure for vertical layout and transparent background
        plt.rcParams['xtick.labelsize'] = 28  # X-axis tick label size

        # Make the figure background transparent
        plt.gcf().set_facecolor('none')

        # L2 Error plot (top)
        plt.subplot(2, 1, 1)
        sns.kdeplot(l2_errors, color='dodgerblue', shade=True, alpha=0.3, log_scale=True)
        plt.axvline(mean_l2, color='black', linestyle='--', linewidth=5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
        plt.yticks([])
        plt.ylabel('')

        # Absolute Error plot (bottom)
        plt.subplot(2, 1, 2)
        sns.kdeplot(abs_errors, color='violet', shade=True, alpha=0.3, log_scale=True)
        plt.axvline(mean_abs, color='black', linestyle='--', linewidth=5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
        plt.yticks([])
        plt.ylabel('')

        plt.tight_layout()

        output_path = '.\\result'
        plot_filename = os.path.join(output_path, 'Tumourpretraindecodervis.svg')
        plt.savefig(plot_filename, bbox_inches='tight', transparent=True)
        plt.show()
def vis_dis(embed_dim=64, seed=420, load_path=None,size = (10,4)):
    setup_seed(seed)
    m = Model(embed_dim=embed_dim, num_points=1000)
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}. Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    l2_errors = []
    abs_errors = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            test_points = batch.processed_contour.to(device)
            ref_points_test = batch.circle_points.to(device)
            fx_test = batch.fx.to(device)
            init_domain_test = batch.initial_domain.to(device)
            x_boundary_test, y_boundary_test, t_boundary_test = ref_points_test[:, :, 0], ref_points_test[:, :,
                                                                                          1], ref_points_test[:, :, 2]
            x_pred_boundary_test, y_pred_boundary_test,x_de,y_de = m.forward(x=x_boundary_test, y=y_boundary_test,
                                                                         t=t_boundary_test, fx=fx_test,
                                                                         init_domain=init_domain_test)
            pred_boundary_test = torch.stack([x_pred_boundary_test, y_pred_boundary_test], dim=2).reshape(-1, 15, 300,2)
            test_points = test_points.reshape(-1, 15, 300, 2)
            batch_l2_errors = torch.norm(test_points.reshape(test_points.shape[0],-1) - pred_boundary_test.reshape(test_points.shape[0],-1), p=2,dim =1) / torch.norm(pred_boundary_test.reshape(test_points.shape[0],-1),dim =1,p = 2)
            batch_abs_errors = torch.mean((test_points.reshape(test_points.shape[0],-1) - pred_boundary_test.reshape(test_points.shape[0],-1))**2, dim=1)

            l2_errors.extend(batch_l2_errors.cpu().numpy())
            abs_errors.extend(batch_abs_errors.cpu().numpy())
        l2_errors = np.array(l2_errors)
        abs_errors = np.array(abs_errors)

        # Calculate mean errors
        mean_l2 = np.mean(l2_errors)
        mean_abs = np.mean(abs_errors)
        print(f"Mean L2 error: {mean_l2}")
        print(f"Mean Absolute error: {mean_abs}")
        # Plot error distributions using KDE
        plt.figure(figsize=size, facecolor='none')  # Taller figure for vertical layout and transparent background
        plt.rcParams['xtick.labelsize'] = 28  # X-axis tick label size

        # Make the figure background transparent
        plt.gcf().set_facecolor('none')

        # L2 Error plot (top)
        plt.subplot(2, 1, 1)
        sns.kdeplot(l2_errors, color='red', shade=True, alpha=0.3, log_scale=True)
        plt.axvline(mean_l2, color='black', linestyle='--', linewidth=5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
        plt.yticks([])
        plt.ylabel('')

        # Absolute Error plot (bottom)
        plt.subplot(2, 1, 2)
        sns.kdeplot(abs_errors, color='yellow', shade=True, alpha=0.3, log_scale=True)
        plt.axvline(mean_abs, color='black', linestyle='--', linewidth=5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)  # Keep bottom spine for reference
        plt.yticks([])
        plt.ylabel('')

        plt.tight_layout()

        output_path = '.\\result'
        plot_filename = os.path.join(output_path, 'Tumourpretrainvis.svg')
        plt.savefig(plot_filename, bbox_inches='tight', transparent=True )
        plt.show()
def absval(embed_dim = 64,seed = 420, load_path=None):
    setup_seed(seed)
    m = Model(embed_dim = embed_dim,num_points=1000)
    if load_path:
        try:
            m.load_state_dict(torch.load(load_path))
            print(f"Model parameters loaded from {load_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {load_path}. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}.  Check model architecture or compatibility. Training from scratch.")
    m = m.to('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    MSE_loss_test = 0
    test_sup = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            target_boundary_test = batch.feq.to(device)
            test_points = batch.processed_contour.to(device)
            ref_points_test = batch.circle_points.to(device)
            fx_test = batch.fx.to(device)
            init_domain_test = batch.initial_domain.to(device)
            x_boundary_test, y_boundary_test,t_boundary_test = ref_points_test[:,:, 0], ref_points_test[:, :,1],ref_points_test[:,:,2]
            x_pred_boundary_test, y_pred_boundary_test,_,_ = m.forward(x=x_boundary_test, y=y_boundary_test, t=t_boundary_test, fx=fx_test,init_domain=init_domain_test)
            pred_boundary_test = torch.stack([x_pred_boundary_test, y_pred_boundary_test], dim=2).reshape(-1,15,300,2)
            test_points = test_points.reshape(-1,15,300,2)
            # feq_loss = fourier_loss(pred_boundary_test,target_boundary_test)
            loss = (pred_boundary_test-test_points).pow(2).sum(dim=-1)
            dis = (pred_boundary_test).pow(2).sum(dim=-1)
            # loss = loss/torch.abs(test_points).pow(2).sum(dim=-1)
            pred = pred_boundary_test.squeeze(0).cpu().numpy()
            target = test_points.squeeze(0).cpu().numpy()
            target_dis=(test_points).pow(2).sum(dim=-1)
            loss = loss**0.5
            dis = dis**0.5
            target_dis = target_dis**0.5
            loss = loss.squeeze(0).cpu().numpy()
            dis = dis.squeeze(0).cpu().numpy()
            target_dis = target_dis.squeeze(0).cpu().numpy()
            if batch_idx <= 2:
                plot_3d_interpolation(target,target_dis,batch_idx=batch_idx,name='target')
                plot_3d_interpolation(pred,dis,batch_idx=batch_idx,name='pred')
                plot_3d_interpolation(pred,loss,batch_idx=batch_idx,name='loss')
            else:
                break

if __name__ == '__main__':

    load_path = ".\\pretrainmodel_posttrained.pth"
    vis_dis(embed_dim=256, load_path=load_path)
