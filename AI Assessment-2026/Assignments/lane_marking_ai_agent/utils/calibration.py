"""
Camera Calibration Utility for Lane Detection
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import yaml


class CameraCalibrator:
    """Camera calibration for lane detection systems."""
    
    def __init__(self, camera_config: dict = None):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography = None
        if camera_config:
            self.load_from_config(camera_config)
    
    def calibrate_intrinsic(self, images: List[np.ndarray], chessboard_size: Tuple = (9, 6), square_size: float = 0.025):
        """Calibrate camera intrinsic parameters using chessboard pattern."""
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints, imgpoints = [], []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        print(f"Calibration RMS error: {ret:.4f}")
        return ret
    
    def load_from_config(self, config: dict):
        if 'camera_matrix' in config:
            self.camera_matrix = np.array(config['camera_matrix'])
        if 'dist_coeffs' in config:
            self.dist_coeffs = np.array(config['dist_coeffs'])
        if 'homography' in config:
            self.homography = np.array(config['homography'])
    
    def save_to_file(self, filepath: str):
        data = {
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            'homography': self.homography.tolist() if self.homography is not None else None
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f)
        print(f"Calibration saved to: {filepath}")