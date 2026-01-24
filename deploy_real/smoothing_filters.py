#!/usr/bin/env python3
"""
Smoothing Filters for Teleoperation.

Provides various smoothing approaches for reducing jitter in motion capture
and robot control. Designed to be applied at different stages:
- Skeleton level (filter MediaPipe noise)
- End-effector level (smooth IK targets)
- Joint angle level (smooth robot output)

Available filters:
- ExponentialMovingAverage (EMA): Simple, one parameter
- OneEuroFilter: Adaptive, smooth when slow, responsive when fast
- VelocityLimiter: Hard clamp on rate of change

Usage:
    from smoothing_filters import EMAFilter, OneEuroFilter, VelocityLimiter
    
    # EMA on joint angles
    ema = EMAFilter(alpha=0.3, num_dims=29)
    smoothed = ema.filter(joint_angles)
    
    # One Euro on skeleton positions  
    one_euro = OneEuroFilter(min_cutoff=1.0, beta=0.01, num_dims=99)
    smoothed = one_euro.filter(skeleton.flatten())
    
    # Velocity limiting for safety
    limiter = VelocityLimiter(max_velocity=2.0, dt=1/30)
    safe_angles = limiter.filter(joint_angles)
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass


class EMAFilter:
    """
    Exponential Moving Average filter.
    
    Simple and effective low-pass filter with one tuning parameter.
    
    smoothed = alpha * new_value + (1 - alpha) * prev_smoothed
    
    Alpha close to 1.0 = more responsive, less smooth
    Alpha close to 0.0 = more smooth, more latency
    
    Recommended: alpha = 0.2-0.4 for motion capture
    """
    
    def __init__(self, alpha: float = 0.3, num_dims: int = None):
        """
        Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
            num_dims: Number of dimensions (optional, for pre-allocation)
        """
        self.alpha = np.clip(alpha, 0.01, 1.0)
        self.num_dims = num_dims
        self.prev: Optional[np.ndarray] = None
        
    def reset(self):
        """Reset filter state."""
        self.prev = None
        
    def filter(self, value: np.ndarray) -> np.ndarray:
        """
        Apply EMA filter to new value.
        
        Args:
            value: New value (any shape)
            
        Returns:
            Smoothed value (same shape)
        """
        value = np.asarray(value, dtype=np.float64)
        
        if self.prev is None:
            self.prev = value.copy()
            return value
        
        smoothed = self.alpha * value + (1 - self.alpha) * self.prev
        self.prev = smoothed.copy()
        
        return smoothed


class OneEuroFilter:
    """
    One Euro Filter - adaptive low-pass filter.
    
    Provides smooth filtering when motion is slow, and responsive
    filtering when motion is fast. Widely used in motion capture.
    
    Reference: Casiez et al. (2012) "1€ Filter: A Simple Speed-based 
    Low-pass Filter for Noisy Input in Interactive Systems"
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (smoothness when still)
        beta: Speed coefficient (responsiveness when moving)
        d_cutoff: Cutoff for derivative (usually leave at 1.0)
    
    Recommended starting values:
        min_cutoff = 1.0, beta = 0.01 for motion capture
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
        num_dims: int = None,
    ):
        """
        Initialize One Euro filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            d_cutoff: Derivative cutoff frequency (Hz)
            num_dims: Number of dimensions (optional)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.num_dims = num_dims
        
        self.prev_value: Optional[np.ndarray] = None
        self.prev_derivative: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        
    def reset(self):
        """Reset filter state."""
        self.prev_value = None
        self.prev_derivative = None
        self.prev_time = None
        
    def _alpha(self, cutoff: float, dt: float) -> float:
        """Compute alpha for given cutoff frequency."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def filter(self, value: np.ndarray, timestamp: float = None) -> np.ndarray:
        """
        Apply One Euro filter to new value.
        
        Args:
            value: New value (any shape)
            timestamp: Time in seconds (uses internal counter if None)
            
        Returns:
            Smoothed value (same shape)
        """
        import time
        
        value = np.asarray(value, dtype=np.float64)
        
        if timestamp is None:
            timestamp = time.time()
        
        if self.prev_value is None:
            self.prev_value = value.copy()
            self.prev_derivative = np.zeros_like(value)
            self.prev_time = timestamp
            return value
        
        # Compute dt
        dt = timestamp - self.prev_time
        if dt <= 0:
            dt = 1/30  # Default to 30 FPS
        self.prev_time = timestamp
        
        # Compute derivative
        derivative = (value - self.prev_value) / dt
        
        # Filter derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        derivative_filtered = alpha_d * derivative + (1 - alpha_d) * self.prev_derivative
        self.prev_derivative = derivative_filtered
        
        # Compute adaptive cutoff based on speed
        speed = np.abs(derivative_filtered)
        cutoff = self.min_cutoff + self.beta * speed
        
        # Filter value with adaptive cutoff
        alpha = self._alpha(cutoff, dt)
        
        # Handle array case - use element-wise alpha
        if isinstance(alpha, np.ndarray):
            smoothed = alpha * value + (1 - alpha) * self.prev_value
        else:
            smoothed = alpha * value + (1 - alpha) * self.prev_value
        
        self.prev_value = smoothed.copy()
        
        return smoothed


class VelocityLimiter:
    """
    Velocity limiting filter for safety.
    
    Clamps the rate of change of values to a maximum velocity.
    Guarantees smooth motion within physical limits.
    
    Critical for robot safety - prevents sudden jumps that could
    cause falls or damage.
    """
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        dt: float = 1/30,
        num_dims: int = None,
    ):
        """
        Initialize velocity limiter.
        
        Args:
            max_velocity: Maximum allowed velocity (units per second)
            dt: Time step (seconds)
            num_dims: Number of dimensions (optional)
        """
        self.max_velocity = max_velocity
        self.dt = dt
        self.num_dims = num_dims
        self.prev: Optional[np.ndarray] = None
        
    def reset(self):
        """Reset filter state."""
        self.prev = None
        
    def filter(self, value: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Apply velocity limiting to new value.
        
        Args:
            value: Target value (any shape)
            dt: Optional override for time step
            
        Returns:
            Velocity-limited value (same shape)
        """
        value = np.asarray(value, dtype=np.float64)
        
        if dt is None:
            dt = self.dt
        
        if self.prev is None:
            self.prev = value.copy()
            return value
        
        # Compute desired change
        delta = value - self.prev
        
        # Compute max allowed change this timestep
        max_delta = self.max_velocity * dt
        
        # Clamp each dimension independently
        delta_clamped = np.clip(delta, -max_delta, max_delta)
        
        # Apply clamped delta
        result = self.prev + delta_clamped
        self.prev = result.copy()
        
        return result


@dataclass
class SmoothingConfig:
    """Configuration for smoothing pipeline."""
    
    method: str = "none"  # "none", "ema", "one_euro", "velocity_limit", "combined"
    
    # EMA parameters
    ema_alpha: float = 0.3
    
    # One Euro parameters  
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.01
    
    # Velocity limiter parameters
    max_velocity: float = 3.0  # rad/s for joints
    
    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return {
            "method": self.method,
            "ema_alpha": self.ema_alpha,
            "one_euro_min_cutoff": self.one_euro_min_cutoff,
            "one_euro_beta": self.one_euro_beta,
            "max_velocity": self.max_velocity,
        }


class SmoothingPipeline:
    """
    Combined smoothing pipeline for teleoperation.
    
    Applies smoothing at a specified stage (skeleton, IK targets, or joints).
    Configurable via SmoothingConfig.
    """
    
    def __init__(self, config: SmoothingConfig, num_dims: int, fps: float = 30):
        """
        Initialize smoothing pipeline.
        
        Args:
            config: Smoothing configuration
            num_dims: Number of dimensions to filter
            fps: Frame rate for velocity limiting
        """
        self.config = config
        self.num_dims = num_dims
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Initialize filters based on config
        self.filters = []
        
        if config.method == "ema":
            self.filters.append(EMAFilter(alpha=config.ema_alpha, num_dims=num_dims))
            
        elif config.method == "one_euro":
            self.filters.append(OneEuroFilter(
                min_cutoff=config.one_euro_min_cutoff,
                beta=config.one_euro_beta,
                num_dims=num_dims,
            ))
            
        elif config.method == "velocity_limit":
            self.filters.append(VelocityLimiter(
                max_velocity=config.max_velocity,
                dt=self.dt,
                num_dims=num_dims,
            ))
            
        elif config.method == "combined":
            # Apply One Euro first, then velocity limiting for safety
            self.filters.append(OneEuroFilter(
                min_cutoff=config.one_euro_min_cutoff,
                beta=config.one_euro_beta,
                num_dims=num_dims,
            ))
            self.filters.append(VelocityLimiter(
                max_velocity=config.max_velocity,
                dt=self.dt,
                num_dims=num_dims,
            ))
    
    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()
    
    def filter(self, value: np.ndarray, timestamp: float = None) -> np.ndarray:
        """
        Apply smoothing pipeline.
        
        Args:
            value: Input value
            timestamp: Optional timestamp for One Euro filter
            
        Returns:
            Smoothed value
        """
        if self.config.method == "none":
            return value
        
        result = value
        for f in self.filters:
            if isinstance(f, OneEuroFilter):
                result = f.filter(result, timestamp)
            else:
                result = f.filter(result)
        
        return result


def demo():
    """Demonstrate smoothing filters on synthetic noisy signal."""
    import time
    
    print("Smoothing Filter Demo")
    print("=" * 50)
    
    # Generate noisy sine wave
    np.random.seed(42)
    t = np.linspace(0, 2, 60)  # 2 seconds at 30 FPS
    clean = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine
    noise = np.random.randn(len(t)) * 0.2
    noisy = clean + noise
    
    # Apply filters
    ema = EMAFilter(alpha=0.3)
    one_euro = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    vel_limit = VelocityLimiter(max_velocity=2.0, dt=1/30)
    
    ema_output = []
    one_euro_output = []
    vel_limit_output = []
    
    for i, val in enumerate(noisy):
        ema_output.append(ema.filter(np.array([val]))[0])
        one_euro_output.append(one_euro.filter(np.array([val]), timestamp=t[i])[0])
        vel_limit_output.append(vel_limit.filter(np.array([val]))[0])
    
    # Compute errors
    def rmse(a, b):
        return np.sqrt(np.mean((np.array(a) - np.array(b))**2))
    
    print(f"\nRMSE from clean signal:")
    print(f"  Noisy:         {rmse(noisy, clean):.4f}")
    print(f"  EMA:           {rmse(ema_output, clean):.4f}")
    print(f"  One Euro:      {rmse(one_euro_output, clean):.4f}")
    print(f"  Vel Limit:     {rmse(vel_limit_output, clean):.4f}")
    
    # Compute velocity variance (jitter measure)
    def vel_var(signal):
        vel = np.diff(signal) / (1/30)
        return np.var(vel)
    
    print(f"\nVelocity Variance (jitter):")
    print(f"  Clean:         {vel_var(clean):.4f}")
    print(f"  Noisy:         {vel_var(noisy):.4f}")
    print(f"  EMA:           {vel_var(ema_output):.4f}")
    print(f"  One Euro:      {vel_var(one_euro_output):.4f}")
    print(f"  Vel Limit:     {vel_var(vel_limit_output):.4f}")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(t, clean, 'k--', label='Clean', linewidth=2)
        plt.plot(t, noisy, 'gray', alpha=0.5, label='Noisy')
        plt.plot(t, ema_output, 'b-', label='EMA', linewidth=1.5)
        plt.plot(t, one_euro_output, 'r-', label='One Euro', linewidth=1.5)
        plt.plot(t, vel_limit_output, 'g-', label='Vel Limit', linewidth=1.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Smoothing Filter Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\n(matplotlib not available for plotting)")


if __name__ == "__main__":
    demo()
