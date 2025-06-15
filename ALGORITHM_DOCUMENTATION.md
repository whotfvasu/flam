# Algorithm Documentation

## Task 1: Capturing and Preparing the Person's Image

### Step 1: Capture a High-Quality Image
- Used a high-quality front-view image of a person.
- Algorithm uses U^2-Net deep learning model to handle various lighting conditions.

### Step 2: Remove the Background
- Implemented advanced background removal using `rembg` library with U^2-Net.
- Used alpha matting for precise edge detection and preservation.
- Created a binary mask for the person to be used in shadow generation.

## Task 2: Analyzing Shadows and Lighting of the Background Image

### Step 1: Detect and Classify Shadows
- Analyzed the scene using multiple color spaces (LAB, HSV) for shadow detection.
- Implemented adaptive thresholding to identify shadows.
- Generated binary masks for detected shadows.
- Distinguished between hard and soft shadows using morphological operations.

## Task 3: Determining Light Direction

### Step 1: Compute Light Direction
- Analyzed bright areas in the scene to detect light sources.
- Used shadow direction to calculate light angle.
- Created a depth map to understand scene perspectives.

### Step 2: Estimate Lighting for Indoor Scenes
- Calculated average brightness and color temperature.
- Analyzed diffused lighting pattern.
- Determined indoor hallway has overhead lighting with moderate diffusion.

## Task 4: Coloring and Blending

### Missing Steps Identified and Implemented
1. **Color Space Transformation**: Converted to LAB color space for better light and color control.
2. **Luminance Adaptation**: Adjusted brightness levels to match scene lighting conditions.
3. **Color Temperature Adjustment**: Added warmth/coolness to match indoor lighting.
4. **Contrast Adjustment**: Modified contrast to match the indoor lighting diffusion.
5. **Texture Harmonization**: Added subtle grunge texture for integration with the scene.

## Task 5: Generating the Final Output

### Process Overview
1. **Position Determination**: Identified proper placement based on scene geometry.
2. **Perspective Matching**: Scaled the person according to scene depth.
3. **Shadow Projection**: Created perspective-correct shadows on the floor.
4. **Alpha Blending**: Used alpha compositing for realistic integration.
5. **Final Color Grading**: Applied cinematic color grading for visual harmony.
6. **Detail Enhancement**: Added subtle sharpening and grain for realism.

## Techniques Used

1. **Advanced Image Segmentation**: U^2-Net for accurate foreground extraction
2. **Multiple Color Space Analysis**: LAB, HSV for better lighting and shadow detection
3. **Depth Mapping**: Created perspective-aware depth maps
4. **Intrinsic Image Decomposition**: Separated lighting from surface reflectance
5. **Alpha Compositing**: Used proper alpha blending for realistic edges
6. **Color Grading**: Applied professional color grading techniques for visual consistency

## Challenges and Solutions

1. **Challenge**: Matching indoor lighting from outdoor-lit person image.
   **Solution**: Used LAB color space adjustments and selective contrast reduction.

2. **Challenge**: Creating realistic shadows for the floor.
   **Solution**: Used perspective transform and depth-based fading.

3. **Challenge**: Avoiding brightness loss during integration.
   **Solution**: Calibrated brightness factors and implemented compensating adjustments.

4. **Challenge**: Achieving realistic placement in the hallway.
   **Solution**: Used floor line detection and scale factor calculation based on scene depth.

## Final Results

The algorithm successfully integrates the person into the background scene with:
- Appropriate positioning and scaling
- Realistic shadows that match the scene's lighting
- Color harmonization between person and background
- Natural blending with the environment

The final composite appears photorealistic with proper lighting, shadows, and color coherence.
