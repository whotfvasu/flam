import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import matplotlib.pyplot as plt
from skimage import exposure, color, transform

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def remove_background(input_path, output_path):
    print(f"Removing background from {input_path}...")
    
    try:
        from rembg import remove, new_session
        session = new_session("u2net_human_seg")
        
        with open(input_path, "rb") as inp_file:
            input_image = inp_file.read()
        
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        with open(output_path, "wb") as out_file:
            out_file.write(output_image)
        
        print(f"Background removed and saved to {output_path}")
        
        foreground = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        return foreground
        
    except Exception as e:
        print(f"Error in background removal: {e}")
        from rembg import remove
        with open(input_path, "rb") as inp_file:
            input_image = inp_file.read()
        output_image = remove(input_image)
        with open(output_path, "wb") as out_file:
            out_file.write(output_image)
        foreground = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
        return foreground

def analyze_scene_depth(bg_path):
    print("Analyzing scene depth...")
    
    bg_image = cv2.imread(bg_path)
    h, w = bg_image.shape[:2]
    
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        depth_value = 1.0 - (y / h) * 0.8
        depth_map[y, :] = depth_value
        
    depth_vis = (depth_map * 255).astype(np.uint8)
    cv2.imwrite("../analysis/depth_map.png", depth_vis)
    
    floor_line = int(h * 0.95)
    
    return {"depth_map": depth_map, "floor_line": floor_line}

def analyze_lighting_conditions(bg_path, output_path):
    print(f"Analyzing lighting conditions in {bg_path}...")
    
    bg_image = cv2.imread(bg_path)
    
    hsv_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:,:,2]
    
    avg_brightness = np.mean(v_channel)
    
    _, bright_areas = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)
    _, dark_areas = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY_INV)
    
    light_angle = 90
    
    contours, _ = cv2.findContours(bright_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis_image = bg_image.copy()
    
    if len(contours) > 0:
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    
    lighting_info = {
        'avg_brightness': avg_brightness,
        'dominant_light_angle': light_angle,
        'light_type': 'indoor_overhead',
        'light_intensity': avg_brightness / 255.0,
        'light_color': (255, 255, 255)
    }
    
    cv2.putText(
        vis_image, 
        f"Avg Brightness: {avg_brightness:.1f}",
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 255), 
        2
    )
    
    cv2.imwrite(output_path, vis_image)
    print(f"Lighting analysis saved to {output_path}")
    
    return lighting_info

def apply_grunge_texture(image, strength=0.3):
    print("Applying grunge texture...")
    
    h, w = image.shape[:2]
    
    noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
    
    noise = cv2.GaussianBlur(noise, (21, 21), 0)
    
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    img_float = image.astype(np.float32) / 255.0
    
    for c in range(3):
        img_float[:,:,c] = img_float[:,:,c] * (1 - strength + strength * noise)
    
    result = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    return result

def match_colors_for_scene(person_img, bg_img, lighting_info, output_path):
    print("Matching colors for indoor hallway scene...")
    
    if person_img.shape[2] == 4:
        alpha = person_img[:,:,3].copy()
        person_rgb = person_img[:,:,:3].copy()
    else:
        person_rgb = person_img.copy()
        alpha = None
    
    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(person_lab)
    
    l_float = l.astype(np.float32)
    l_mean = np.mean(l_float)
    l_adjusted = l_mean + (l_float - l_mean) * 0.9
    l_adjusted = l_adjusted * 1.15
    l = np.clip(l_adjusted, 0, 255).astype(np.uint8)
    
    a = np.clip(a.astype(np.float32) + 2, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) + 3, 0, 255).astype(np.uint8)
    
    person_lab_adjusted = cv2.merge([l, a, b])
    person_rgb_adjusted = cv2.cvtColor(person_lab_adjusted, cv2.COLOR_LAB2BGR)
    
    light_intensity = lighting_info['light_intensity']
    
    brightness_factor = 1.0
    person_rgb_adjusted = cv2.convertScaleAbs(person_rgb_adjusted, alpha=brightness_factor, beta=5)
    
    h, w = person_rgb_adjusted.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h//2, w//2
    mask = 1 - 0.1 * np.sqrt(((x - center_x)/(w/2))**2 + ((y - center_y)/(h/2))**2)
    mask = np.clip(mask, 0, 1)
    
    mask_3channel = np.ones_like(person_rgb_adjusted, dtype=np.float32)
    for c in range(3):
        mask_3channel[:,:,c] = mask
    
    person_rgb_adjusted = (person_rgb_adjusted.astype(np.float32) * mask_3channel).astype(np.uint8)
    
    if alpha is not None:
        person_rgba = cv2.merge([person_rgb_adjusted[:,:,0], 
                               person_rgb_adjusted[:,:,1], 
                               person_rgb_adjusted[:,:,2], 
                               alpha])
        cv2.imwrite(output_path, person_rgba)
        result = person_rgba
    else:
        cv2.imwrite(output_path, person_rgb_adjusted)
        result = person_rgb_adjusted
        
    print(f"Color-matched image saved to {output_path}")
    return result

def generate_appropriate_shadow(person_mask, scene_depth, lighting_info, output_path):
    print("Generating appropriate shadow for hallway scene...")
    
    h, w = person_mask.shape[:2]
    
    shadow = np.zeros((h, w), dtype=np.float32)
    
    mask = person_mask.copy().astype(np.float32) / 255.0
    
    floor_line = scene_depth['floor_line']
    perspective_pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    perspective_pts2 = np.float32([[w*0.1, 0], [w*0.9, 0], [0, h], [w, h]])
    
    M = cv2.getPerspectiveTransform(perspective_pts1, perspective_pts2)
    mask_warped = cv2.warpPerspective(mask, M, (w, h))
    
    for y in range(h):
        if y >= floor_line:
            shadow[y, :] = mask_warped[y, :] * 0.4
    
    shadow = cv2.GaussianBlur(shadow, (21, 21), 0)
    
    depth_map = scene_depth['depth_map']
    if depth_map.shape != shadow.shape:
        print(f"Resizing depth map from {depth_map.shape} to {shadow.shape}")
        depth_map_resized = cv2.resize(depth_map, (w, h))
    else:
        depth_map_resized = depth_map
    
    shadow = shadow * (1.0 - depth_map_resized * 0.7)
    
    cv2.imwrite(output_path, shadow * 255)
    print(f"Scene-appropriate shadow saved to {output_path}")
    
    return shadow

def blend_for_hallway(person_img, bg_img, shadow, scene_depth, output_path):
    print("Blending person into hallway scene...")
    
    result = bg_img.copy()
    bg_h, bg_w = bg_img.shape[:2]
    
    fg_h, fg_w = person_img.shape[:2]
    
    scale_factor = 0.8
    
    new_height = int(fg_h * scale_factor)
    new_width = int(fg_w * scale_factor)
    person_resized = cv2.resize(person_img, (new_width, new_height))
    
    floor_line = scene_depth['floor_line']
    
    x_offset = bg_w // 2 - new_width // 2
    y_offset = floor_line - new_height
    
    x_offset = max(0, min(x_offset, bg_w - new_width))
    y_offset = max(0, min(y_offset, bg_h - new_height))
    
    shadow_resized = cv2.resize(shadow, (new_width, new_height))
    
    for y in range(new_height):
        for x in range(new_width):
            bg_x = x + x_offset
            bg_y = y + y_offset
            
            if 0 <= bg_x < bg_w and 0 <= bg_y < bg_h:
                shadow_alpha = shadow_resized[y, x]
                if shadow_alpha > 0:
                    for c in range(3):
                        result[bg_y, bg_x, c] = int(result[bg_y, bg_x, c] * (1 - shadow_alpha * 0.3))
    
    for y in range(new_height):
        for x in range(new_width):
            bg_x = x + x_offset
            bg_y = y + y_offset
            
            if 0 <= bg_x < bg_w and 0 <= bg_y < bg_h:
                if person_resized.shape[2] == 4:
                    alpha = person_resized[y, x, 3] / 255.0
                else:
                    alpha = 1.0
                
                if alpha > 0.05:
                    for c in range(3):
                        result[bg_y, bg_x, c] = int(result[bg_y, bg_x, c] * (1 - alpha) + 
                                                  person_resized[y, x, c] * alpha)
    
    result = apply_grunge_texture(result, strength=0.1)
    
    cv2.imwrite(output_path, result)
    print(f"Final hallway composite saved to {output_path}")
    
    return result

def apply_final_adjustments(image_path, output_path):
    print("Applying final adjustments...")
    
    image = cv2.imread(image_path)
    
    image = cv2.convertScaleAbs(image, alpha=1.15, beta=10)
    
    h, w = image.shape[:2]
    grain = np.random.normal(0, 1, (h, w, 3)).astype(np.int32)
    image_with_grain = np.clip(image.astype(np.int32) + grain, 0, 255).astype(np.uint8)
    
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]]) / 2.0
    sharpened = cv2.filter2D(image_with_grain, -1, kernel)
    
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_float = l.astype(np.float32)
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)
    
    shadow_mask = (l_float < 100).astype(np.float32)
    b_float = b_float - shadow_mask * 1
    
    highlight_mask = (l_float > 150).astype(np.float32)
    b_float = b_float + highlight_mask * 2
    
    a = np.clip(a_float, 0, 255).astype(np.uint8)
    b = np.clip(b_float, 0, 255).astype(np.uint8)
    lab_adjusted = cv2.merge([l, a, b])
    
    final = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    rows, cols = final.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/4)
    kernel_y = cv2.getGaussianKernel(rows, rows/4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    for i in range(3):
        final[:,:,i] = final[:,:,i] * (0.98 + 0.02 * mask)
    
    cv2.imwrite(output_path, final)
    print(f"Final adjustments saved to {output_path}")
    
    return final

def main():
    create_directory_if_not_exists("../results")
    create_directory_if_not_exists("../masks")
    create_directory_if_not_exists("../analysis")
    
    person_path = "../images/person.jpg"
    bg_path = "../images/bg2.jpg"
    
    person_no_bg_path = "../results/person_no_bg.png"
    person_no_bg = remove_background(person_path, person_no_bg_path)
    
    if person_no_bg.shape[2] == 4:
        person_alpha = person_no_bg[:,:,3]
    else:
        person_gray = cv2.cvtColor(person_no_bg, cv2.COLOR_BGR2GRAY)
        _, person_alpha = cv2.threshold(person_gray, 10, 255, cv2.THRESH_BINARY)
        
    person_mask_path = "../masks/person_mask.png"
    cv2.imwrite(person_mask_path, person_alpha)
    
    scene_depth = analyze_scene_depth(bg_path)
    
    bg_image = cv2.imread(bg_path)
    lighting_analysis_path = "../analysis/lighting_analysis.png"
    lighting_info = analyze_lighting_conditions(bg_path, lighting_analysis_path)
    
    color_matched_path = "../results/person_color_matched.png"
    color_matched = match_colors_for_scene(person_no_bg, bg_image, lighting_info, color_matched_path)
    
    shadow_path = "../masks/person_shadow.png"
    shadow = generate_appropriate_shadow(person_alpha, scene_depth, lighting_info, shadow_path)
    
    composite_path = "../results/composite.png"
    composite = blend_for_hallway(color_matched, bg_image, shadow, scene_depth, composite_path)
    
    final_path = "../results/final_output.png"
    final_image = apply_final_adjustments(composite_path, final_path)
    
    print("Processing complete!")
    print(f"Final output saved to {final_path}")

if __name__ == "__main__":
    main()
