# AirSim Camera and IMU Calibration Script

This Python script connects to an AirSim simulation, retrieves camera intrinsics, image dimensions, and IMU noise characteristics. The script uses the camera's field of view (FOV) to calculate the focal lengths and principal points, and it estimates noise characteristics for the IMU's gyroscope and accelerometer.

## Formulas Used

### 1. **Focal Length Calculation**
   To calculate the focal lengths (`fx` and `fy`) based on the FOV and the image dimensions:

   - **Formula:**
     \[
     f_x = \frac{\text{width} / 2}{\tan(\text{FOV} / 2)}
     \]
     \[
     f_y = \frac{\text{height} / 2}{\tan(\text{FOV} / 2)}
     \]

   - **Explanation**: The focal length can be calculated using the width (or height) of the image and the field of view angle. This is derived from the pinhole camera model, where the focal length is the distance from the camera center to the image plane.

   - **Parameters**:
     - `width`: Image width in pixels
     - `height`: Image height in pixels
     - `FOV`: Field of view in degrees (converted to radians within the script)

### 2. **Principal Point Calculation**
   The principal points (`cx` and `cy`) are assumed to be at the center of the image:

   - **Formula:**
     \[
     c_x = \frac{\text{width}}{2}
     \]
     \[
     c_y = \frac{\text{height}}{2}
     \]

   - **Explanation**: In a standard pinhole camera model, the principal point is generally at the center of the image. 

### 3. **IMU Noise Estimation**
   To estimate the noise characteristics of the IMU’s gyroscope and accelerometer, the standard deviation is calculated for each axis (x, y, z) based on multiple samples.

   - **Formula**:
     \[
     \sigma_{\text{gyro\_axis}} = \text{std}([g_x, g_y, g_z])
     \]
     \[
     \sigma_{\text{acc\_axis}} = \text{std}([a_x, a_y, a_z])
     \]

   - **Explanation**: The standard deviation of the samples from each axis (x, y, z) gives a measure of the noise level for the gyroscope and accelerometer.

   - **Parameters**:
     - `g_x, g_y, g_z`: Gyroscope readings on x, y, and z axes.
     - `a_x, a_y, a_z`: Accelerometer readings on x, y, and z axes.

## Output

The script will output:
- Camera intrinsics: `fx`, `fy`, `cx`, and `cy`.
- Image dimensions: `width` and `height`.
- Estimated IMU noise for the gyroscope and accelerometer.
