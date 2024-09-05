# import architectures from local defintions and external packages
import segmentation_models_pytorch as smp
import torch


# Define a quickloader function
def create_model(
    architecture,
    in_channels,
    num_classes,
    activation=None,
    encoder_name="resnet34",
    encoder_weights="imagenet",
    **kwargs,
):
    """
    Create a segmentation model with the specified architecture.

    This function serves as a wrapper to create various segmentation model architectures
    using the segmentation_models_pytorch (smp) library.

    Args:
        architecture (str): The name of the architecture to use (e.g., 'unet', 'unetplusplus', 'FPN', etc.).
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        activation (str, optional): Activation function to use. Defaults to None.
        encoder_name (str, optional): Name of the encoder to use. Defaults to 'resnet34'.
        encoder_weights (str, optional): Pre-trained weights for the encoder. Defaults to 'imagenet'.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: The created segmentation model.

    Raises:
        ValueError: If an unsupported architecture is specified.
    """
    if architecture == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "Linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "PAN":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "MAnet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    elif architecture == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation,
            **kwargs,
        )
    else:
        raise ValueError(f"Architecture is not recognized: {architecture}")
    return model
