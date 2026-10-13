Enterprise Biometric Security & Access Control System
A high-performance, local biometric access control system built with Python, Flask, and OpenCV. It features real-time facial recognition, multi-layered liveness detection (anti-spoofing), dynamic user enrollment, and an interactive admin dashboard.
Key Features:
 Multi-Layer Anti-Spoofing (Liveness Detection): Laplacian Variance, Micro-Texture Analysis, YCrCb Color Spectrum Filtering.
 Facial Recognition Pipeline: Fast template-based face detection and verification using OpenCV, session-bound temporary ROI caching during user enrollment.
 Interactive Face ID Interface: Modern, responsive dark-mode security dashboard.
 Telemetry & Audit Logs: Automated dynamic logging, intruder snapshot capture and storage upon spoof detection.
Tech Stack:
 Backend: Python, Flask
 Computer Vision: OpenCV (cv2)
 Frontend: HTML5, CSS3, JavaScript (Fetch API)
 Data Storage: JSON / Pickle
