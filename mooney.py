import cv2


def make_mooney(input_path, output_path, blur_sigma=3.0, threshold=127):
    # 1. Load the image in greyscale
    grey = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if grey is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # 2. Apply Gaussian blur
    # (0, 0): OpenCV calculates the kernel size from sigma
    blurred = cv2.GaussianBlur(
        grey,
        (0, 0),
        sigmaX=blur_sigma
    )

    # 3. Apply the threshold
    # Pixel > threshold: white; otherwise: black
    _, mooney = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # 4. Save the result as PNG
    if not cv2.imwrite(output_path, mooney):
        raise OSError(f"Cannot save image: {output_path}")

    return mooney


# Change the filenames to match your image
result = make_mooney(
    input_path="000001.jpg",
    output_path="000001_mooney.png",
    blur_sigma=3.0,
    threshold=127
)