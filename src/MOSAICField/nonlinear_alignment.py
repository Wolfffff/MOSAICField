import torch

from MOSAICField import Loss, Utils
from MOSAICField.MIND import mind_loss_2d_multichannel, zscore_batch
from MOSAICField.Network import DisplacementField


def warp_image(image, field, device):
    """
    Warps an image using a displacement field via a Spatial Transformer.

    Args:
        image (torch.Tensor): The input image tensor with shape [1, C, H, W].
        field (torch.Tensor): The displacement field tensor with shape [1, 2, H, W].
        device (torch.device): The device (CPU or CUDA) to perform the computation on.

    Returns:
        torch.Tensor: The warped image tensor with shape [1, C, H, W].
    """
    ST = Utils.SpatialTransformer(field.shape[2:]).to(device)
    warped_image = ST(
        image.to(device).float(), field.to(device), return_phi=False
    )  # [1, C, H, W]
    return warped_image  # [1, C, H, W]


def warp_image_reverse(image, field, device):
    """
    Warp a target image by reversing a displacement field.

    Parameters
    ----------
    image : torch.Tensor, shape [1, C, H, W]
    field : torch.Tensor, shape [1, 2, H, W]
    device : torch.device or str

    Returns
    -------
    warped_image : torch.Tensor, shape [1, C, H, W]
    phi : torch.Tensor, shape [1, 2, H, W]
    """
    ST = Utils.SpatialTransformer(field.shape[2:]).to(device)
    warped_image, phi = ST.backward(
        image.to(device).float(), field.to(device), return_phi=True
    )  # [1, C, H, W]
    return warped_image, phi  # [1, C, H, W], [1, 2, H, W]


def nonlinear_align(config, source, target, hi_res=None, trained_weight_path=None):
    """
    MOSAICField nonlinear alignment via neural displacement field.

    Parameters
    ----------
    config : object
        Runtime settings. Expected attributes:
        - device: torch device string
        - lr: optimizer learning rate
        - epoches: number of training iterations
        - lambda_J: Jacobian regularization weight
        - lambda_v: magnitude regularization weight
    source : torch.Tensor
        Source image, shape [C1, H, W].
    target : torch.Tensor
        Target image, shape [C2, H, W].
    hi_res : tuple(int, int) or None, optional
        If (H_hi, W_hi), evaluate on higher-resolution grid.
    trained_weight_path : str or None, optional
        Path to saved state_dict. If provided, skip training.

    Returns
    -------
    If hi_res is None:
        (warped_source [C1,H,W], pred_field_grid [1,2,H,W],
         network)
    If hi_res is provided:
        (warped_source [C1,H,W], pred_field_grid [1,2,H,W],
         pred_field_grid_hi [1,2,H_hi,W_hi], network)
    """
    H, W = target.shape[1], target.shape[2]
    source = source.to(config.device).float()
    target = target.to(config.device).float()
    print("source.shape", source.shape)
    print("target.shape", target.shape)

    # Preparing grid for neural field
    grid = Utils.generate_grid2D_tensor((H, W)).to(
        config.device
    )  # [2, H, W], values between [-1,1]
    print("grid.shape", grid.shape)
    grid_batch = grid.permute(1, 2, 0).reshape(H * W, 2)  # [H * W, 2]
    print("grid_batch.shape", grid_batch.shape)
    scale_factor = torch.tensor([H, W]).to(config.device)  # [2]
    print("scale_factor.shape", scale_factor.shape)
    ST = Utils.SpatialTransformer([H, W]).to(
        config.device
    )  # spatial transformer to warp image

    # grid for high res
    if hi_res:
        H_hi, W_hi = hi_res
        grid_hi = Utils.generate_grid2D_tensor((H_hi, W_hi)).to(
            "cpu"
        )  # [2, H_hi, W_hi], values between [-1,1]
        grid_batch_hi = grid_hi.permute(1, 2, 0).reshape(
            H_hi * W_hi, 2
        )  # [H_hi * W_hi, 2]
        scale_factor_hi = torch.tensor([H_hi, W_hi]).to("cpu")  # [2]
        # ST_hi = Utils.SpatialTransformer([H_hi, W_hi]).to("cpu")

    # Define field network
    network = DisplacementField(dim=2, hidden_list=[64, 64, 64]).to(config.device)
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params}")

    # Training loop
    if trained_weight_path is not None:
        state_dict = torch.load(trained_weight_path, map_location=config.device)
        network.load_state_dict(state_dict)

        pred_field_batch = network(grid_batch)  # [H * W, 2]
        pred_field_batch = (
            pred_field_batch * scale_factor
        )  # values [-1, 1] -> voxel spacing
        pred_field_grid = pred_field_batch.reshape(H, W, 2).permute(
            2, 0, 1
        )  # [2, H, W]
        warped_source, phi = ST(
            source.unsqueeze(0), pred_field_grid.unsqueeze(0), return_phi=True
        )  # [1, C, H, W], [1, H, W, 2]
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, amsgrad=True)
        for i in range(config.epoches):
            pred_field_batch = network(grid_batch)  # [H * W, 2]
            pred_field_batch = (
                pred_field_batch * scale_factor
            )  # values [-1, 1] -> voxel spacing
            pred_field_grid = pred_field_batch.reshape(H, W, 2).permute(
                2, 0, 1
            )  # [2, H, W]
            warped_source, phi = ST(
                source.unsqueeze(0), pred_field_grid.unsqueeze(0), return_phi=True
            )  # [1, C, H, W], [1, H, W, 2]
            # loss
            loss_mind = mind_loss_2d_multichannel(
                zscore_batch(warped_source), zscore_batch(target.unsqueeze(0))
            )
            loss_mse = torch.nn.functional.mse_loss(
                zscore_batch(warped_source.mean(dim=1, keepdim=True)),
                zscore_batch(target.unsqueeze(0).mean(dim=1, keepdim=True)),
            )

            loss_J = Loss.neg_Jdet_loss(phi)
            loss_v = Loss.magnitude_loss(pred_field_grid.unsqueeze(0))
            loss = (
                0.5 * loss_mse
                + 0.5 * loss_mind
                + config.lambda_J * loss_J
                + config.lambda_v * loss_v
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 500) == 0:
                print("Iteration: {0} Loss: {1:.3e}".format(i + 1, loss.item()))

    network = network.to("cpu")
    if hi_res:
        pred_field_batch_hi = network(grid_batch_hi)  # [H_hi * W_hi, 2]
        pred_field_batch_hi = (
            pred_field_batch_hi * scale_factor_hi
        )  # values [-1, 1] -> voxel spacing
        pred_field_grid_hi = pred_field_batch_hi.reshape(H_hi, W_hi, 2).permute(
            2, 0, 1
        )  # [2, H_hi, W_hi]

        # [C, H, W], [1, 2, H, W], [1, 2, H_hi, W_hi], network
        return (
            warped_source.squeeze(0),
            pred_field_grid.unsqueeze(0),
            pred_field_grid_hi.unsqueeze(0),
            network,
        )
    else:
        # [C, H, W], [1, 2, H, W], network
        return warped_source.squeeze(0), pred_field_grid.unsqueeze(0), network
