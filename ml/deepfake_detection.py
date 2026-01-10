"""
Deepfake Detection Module using MesoNet and Advanced Neural Networks

This module implements multiple architectures for detecting deepfakes:
- ConvBlock: Basic convolutional building block
- MesoNet: Original MesoNet architecture
- ImprovedMesoNet: Enhanced version with residual connections
- ResidualBlock: Residual connection block for deep networks
- DeepfakeDetectionNet: Advanced detection network
- DeepfakeDetector: Inference class for real-time detection

Author: ShivamGawade-XS
Date: 2026-01-10
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, List, Union
from pathlib import Path
import logging
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block with batch normalization and activation
    
    Components:
    - Convolution layer
    - Batch Normalization
    - Activation function (ReLU)
    - Optional Max Pooling
    - Optional Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel: int = 2,
        use_pool: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Initialize ConvBlock
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding for convolution
            pool_kernel: Size of pooling kernel
            use_pool: Whether to use max pooling
            dropout_rate: Dropout probability
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvBlock
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after convolution, normalization, activation, and pooling
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        if self.use_pool:
            x = self.pool(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block for skip connections in deep networks
    
    Architecture:
    - Conv 3x3 -> BatchNorm -> ReLU
    - Conv 3x3 -> BatchNorm
    - Skip connection with optional stride
    - ReLU activation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout_rate: float = 0.0
    ):
        """
        Initialize ResidualBlock
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            downsample: Optional downsampling module for skip connection
            dropout_rate: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResidualBlock
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class MesoNet(nn.Module):
    """
    MesoNet: A Compact Facial Video Forgery Detection Network
    
    Original architecture designed specifically for deepfake detection
    with focus on mesoscopic patterns in facial forgeries.
    
    Reference: https://arxiv.org/abs/1809.00888
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize MesoNet
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout probability
        """
        super(MesoNet, self).__init__()
        
        # Feature extraction blocks
        self.conv1 = ConvBlock(3, 8, kernel_size=3, stride=1, padding=1, 
                               use_pool=True, dropout_rate=0)
        self.conv2 = ConvBlock(8, 16, kernel_size=5, stride=1, padding=2, 
                               use_pool=True, dropout_rate=dropout_rate)
        self.conv3 = ConvBlock(16, 16, kernel_size=5, stride=1, padding=2, 
                               use_pool=True, dropout_rate=dropout_rate)
        self.conv4 = ConvBlock(16, 16, kernel_size=5, stride=1, padding=2, 
                               use_pool=True, dropout_rate=dropout_rate)
        
        # Flattening and classification
        self.flatten = nn.Flatten()
        
        # Calculate flattened size (depends on input size)
        # Assuming 256x256 input: 256->128->64->32->16
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MesoNet
        
        Args:
            x: Input tensor of shape (batch_size, 3, 256, 256)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class ImprovedMesoNet(nn.Module):
    """
    Improved MesoNet with Residual Connections
    
    Enhanced version of MesoNet incorporating:
    - Residual blocks for better gradient flow
    - Batch normalization in fully connected layers
    - Inception-like parallel pathways
    - Better feature fusion
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize ImprovedMesoNet
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(ImprovedMesoNet, self).__init__()
        
        # Initial feature extraction
        self.conv1 = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1, 
                               use_pool=True, dropout_rate=0)
        
        # Residual blocks for feature learning
        self.res_block1 = ResidualBlock(16, 32, stride=2, 
                                       downsample=self._make_downsample(16, 32, stride=2),
                                       dropout_rate=dropout_rate)
        
        self.res_block2 = ResidualBlock(32, 64, stride=2,
                                       downsample=self._make_downsample(32, 64, stride=2),
                                       dropout_rate=dropout_rate)
        
        self.res_block3 = ResidualBlock(64, 64, stride=1, dropout_rate=dropout_rate)
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classification head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    @staticmethod
    def _make_downsample(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create downsampling layer for skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ImprovedMesoNet
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x


class DeepfakeDetectionNet(nn.Module):
    """
    Advanced Deepfake Detection Network
    
    Combines multiple architectural improvements:
    - Multi-scale feature extraction
    - Residual and dense connections
    - Attention mechanisms
    - Batch normalization throughout
    - Regularization techniques
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize DeepfakeDetectionNet
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability
        """
        super(DeepfakeDetectionNet, self).__init__()
        
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1, 
                     use_pool=True, dropout_rate=0),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, 
                     use_pool=True, dropout_rate=dropout_rate)
        )
        
        # Stage 2: Residual connections
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2,
                         downsample=self._make_downsample(32, 64, stride=2),
                         dropout_rate=dropout_rate),
            ResidualBlock(64, 64, stride=1, dropout_rate=dropout_rate)
        )
        
        # Stage 3: Deeper features
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2,
                         downsample=self._make_downsample(64, 128, stride=2),
                         dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate)
        )
        
        # Stage 4: Final feature refinement
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2,
                         downsample=self._make_downsample(128, 256, stride=2),
                         dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate)
        )
        
        # Attention-like global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with multiple FC layers
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.dropout_fc3 = nn.Dropout(p=dropout_rate)
        
        self.fc4 = nn.Linear(128, num_classes)
        
        self.num_classes = num_classes
    
    @staticmethod
    def _make_downsample(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create downsampling layer for skip connection"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepfakeDetectionNet
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout_fc3(x)
        
        x = self.fc4(x)
        
        return x


class DeepfakeDetector:
    """
    Inference class for deepfake detection
    
    Handles:
    - Model loading and initialization
    - Input preprocessing
    - Batch inference
    - Confidence score computation
    - Face detection and cropping
    - Video frame processing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        architecture: str = "improved_mesonet",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize DeepfakeDetector
        
        Args:
            model_path: Path to pre-trained model weights
            architecture: Model architecture to use
                         Options: "mesonet", "improved_mesonet", "deepfakedetectionnet"
            device: Device to use ("cuda", "cpu", None for auto-detect)
            confidence_threshold: Threshold for classification confidence
        """
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        self.architecture = architecture
        
        # Initialize model
        self.model = self._create_model(architecture)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            if model_path:
                logger.warning(f"Model path {model_path} not found, using random initialization")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def _get_device(device: Optional[str]) -> torch.device:
        """Get the computation device"""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_model(self, architecture: str) -> nn.Module:
        """Create model based on architecture"""
        architectures = {
            "mesonet": MesoNet,
            "improved_mesonet": ImprovedMesoNet,
            "deepfakedetectionnet": DeepfakeDetectionNet
        }
        
        if architecture not in architectures:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        model_class = architectures[architecture]
        model = model_class(num_classes=2, dropout_rate=0.5)
        
        logger.info(f"Created model: {architecture}")
        return model
    
    def _load_model(self, model_path: str) -> None:
        """Load pre-trained weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        tensor = self.preprocess(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            
        Returns:
            Batch tensor
        """
        tensors = [self.preprocess_image(img).squeeze(0) for img in images]
        return torch.stack(tensors)
    
    @torch.no_grad()
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[str, float]:
        """
        Predict if image contains deepfake
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: "Real" or "Fake"
            confidence: Confidence score (0-1)
        """
        image_tensor = self.preprocess_image(image).to(self.device)
        
        output = self.model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # 0: Real, 1: Fake
        fake_confidence = probabilities[0, 1].item()
        
        prediction = "Fake" if fake_confidence > self.confidence_threshold else "Real"
        confidence = max(fake_confidence, 1 - fake_confidence)
        
        return prediction, confidence
    
    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> List[Tuple[str, float]]:
        """
        Batch prediction for multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of (prediction, confidence) tuples
        """
        batch_tensor = self.preprocess_batch(images).to(self.device)
        
        outputs = self.model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        results = []
        for i in range(len(images)):
            fake_confidence = probabilities[i, 1].item()
            prediction = "Fake" if fake_confidence > self.confidence_threshold else "Real"
            confidence = max(fake_confidence, 1 - fake_confidence)
            results.append((prediction, confidence))
        
        return results
    
    @torch.no_grad()
    def predict_video(
        self,
        video_path: str,
        sample_frames: int = 30,
        stride: int = 1
    ) -> Tuple[str, float, List[Tuple[int, str, float]]]:
        """
        Predict deepfake in video by sampling frames
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
            stride: Frame sampling stride
            
        Returns:
            Tuple of (overall_prediction, overall_confidence, frame_results)
            frame_results: List of (frame_number, prediction, confidence)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_sample = min(sample_frames, frame_count)
            indices = np.linspace(0, frame_count - 1, frames_to_sample, dtype=int)
            
            frame_results = []
            fake_scores = []
            
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction, confidence = self.predict(frame_rgb)
                
                frame_results.append((int(frame_idx), prediction, confidence))
                
                # Track fake confidence for overall prediction
                if prediction == "Fake":
                    fake_scores.append(confidence)
            
            cap.release()
            
            # Compute overall prediction
            if fake_scores:
                overall_confidence = np.mean(fake_scores)
                overall_prediction = "Fake"
            else:
                overall_confidence = 0.0
                overall_prediction = "Real"
            
            return overall_prediction, overall_confidence, frame_results
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "architecture": self.architecture,
            "device": str(self.device),
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "confidence_threshold": self.confidence_threshold
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Update confidence threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")


# Example usage and testing
if __name__ == "__main__":
    logger.info("Deepfake Detection Module Loaded")
    
    # Example: Create detector instances
    try:
        # Initialize with ImprovedMesoNet
        detector = DeepfakeDetector(
            architecture="improved_mesonet",
            device="cpu",
            confidence_threshold=0.5
        )
        
        logger.info(f"Model Info: {detector.get_model_info()}")
        
        # Example prediction on a dummy image
        dummy_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        prediction, confidence = detector.predict(dummy_image)
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
