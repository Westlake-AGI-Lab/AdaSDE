import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import dnnlib
import tqdm


class InceptionFeatureExtractor:
    """Inception-v3 feature extractor wrapper (StyleGAN3 metric variant).

    This class loads the Inception network used by StyleGAN metrics and exposes
    a simple `extract_features` method that returns 2048-D features for a batch
    of images.

    Attributes:
        device: Torch device on which the model runs.
        detector_url: URL to the pickled Inception network.
        detector_kwargs: Extra kwargs passed into the network's forward call.
        feature_dim: Dimensionality of the returned feature vectors.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.detector_url = (
            "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
            "versions/1/files/metrics/inception-2015-12-05.pkl"
        )
        self.detector_kwargs = dict(return_features=True)
        self.feature_dim = 2048

        # Load the Inception network from the pickle and move it to the target device.
        with dnnlib.util.open_url(self.detector_url) as f:
            self.detector_net = pickle.load(f).to(device)
        self.detector_net.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Compute Inception features for a batch of images.

        Args:
            images: Tensor of shape [B, C, H, W]. If C == 1, the channel will be
                triplicated to match the expected 3-channel input.

        Returns:
            Tensor of shape [B, 2048] containing per-image feature vectors.
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = images.to(self.device)
        features = self.detector_net(images, **self.detector_kwargs)
        return features


def compute_inception_mse_loss(
    student_images: torch.Tensor,
    teacher_images: torch.Tensor,
    feature_extractor: InceptionFeatureExtractor,
) -> torch.Tensor:
    """Compute MSE loss between Inception features of two image batches.

    Args:
        student_images: Images produced by the student model, shape [B, C, H, W].
        teacher_images: Images produced by the teacher model, shape [B, C, H, W].
        feature_extractor: An initialized `InceptionFeatureExtractor`.

    Returns:
        Scalar tensor containing the mean squared error loss.
    """
    # Extract features
    student_features = feature_extractor.extract_features(student_images)
    teacher_features = feature_extractor.extract_features(teacher_images)

    # Compute MSE loss on feature vectors
    loss = F.mse_loss(student_features, teacher_features)
    return loss
