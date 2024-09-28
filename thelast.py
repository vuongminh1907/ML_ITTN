import cv2
import numpy as np
def Pasting(img_origin,img_cloth):

    #resize về ảnh có kích thước 768x1024
    img_origin = cv2.resize(img_origin, (1024, 768))

    print(img_origin.shape)
    print(img_cloth.shape)

    # Kiểm tra kích thước ảnh
    if img_origin.shape != img_cloth.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước.")
    
    # Tạo mặt nạ từ ảnh thứ 2 (giả sử nền trắng)
    gray_cloth = cv2.cvtColor(img_cloth, cv2.COLOR_BGR2GRAY)

    #save gray_cloth
    cv2.imwrite("gray_cloth.jpg", gray_cloth)

    # Tạo mặt nạ từ ảnh grayscale
    _, mask = cv2.threshold(gray_cloth, 240, 255, cv2.THRESH_BINARY_INV)

    # Làm mờ mặt nạ để loại bỏ viền trắng
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    #save mask
    cv2.imwrite("mask.jpg", mask)

    # Đảm bảo rằng mask có kiểu dữ liệu phù hợp
    mask = mask.astype(np.float32) / 255.0

    # Lấy vùng ảnh thứ 2 mà không phải là nền trắng
    img_cloth_masked = cv2.multiply(img_cloth.astype(np.float32), mask[..., None])

    # Lấy vùng tương ứng từ ảnh gốc
    img_origin_masked = cv2.multiply(img_origin.astype(np.float32), 1.0 - mask[..., None])

    # Kết hợp ảnh gốc và ảnh thứ 2
    result = cv2.add(img_origin_masked, img_cloth_masked)
    result = result.astype(np.uint8)

    return result

# Load ảnh gốc và ảnh thứ 2
img_origin = cv2.imread("image.jpg")
img_cloth = cv2.imread("cloth.jpg")

result = Pasting(img_origin, img_cloth)
#save the result
cv2.imwrite("result.jpg", result)