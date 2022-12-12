from skimage import color, io
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_erosion, binary_dilation
from skimage.morphology import disk
from skimage import segmentation
from skimage import measure
import math
from scipy.stats import norm


def pca_on_football_data():
    in_dir = "data/"
    txt_name = "soccer_data.txt"

    soccer_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = soccer_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    mn = np.mean(x, axis=0)
    data = x - mn
    c_x = np.cov(data.T)

    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    # Project data
    pc_proj = vectors.T.dot(data.T)
    min_val = pc_proj.min()
    max_val = pc_proj.max()
    answer = max(np.abs(min_val), np.abs(max_val))
    print(f"Answer: {answer:.2f}")


def car_rgb_hsv_thresholds():
    in_dir = "data/"
    im_name = "car.png"
    im_org = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_org)
    # hue_img = hsv_img[:, :, 0]
    # value_img = hsv_img[:, :, 2]
    s_img = hsv_img[:, :, 1]

    segm_car = (s_img > 0.7)
    io.imshow(segm_car)
    plt.title('Segmented car')
    io.show()

    print(f"Segm Result {segm_car.sum()}")

    footprint = disk(6)
    eroded = erosion(segm_car, footprint)

    footprint = disk(4)
    dilated = dilation(eroded, footprint)

    result = dilated.sum()
    print(f"Result {result}")

    footprint = disk(6)
    eroded = binary_erosion(segm_car, footprint)

    footprint = disk(4)
    dilated = binary_dilation(eroded, footprint)

    result = dilated.sum()
    print(f"Result 2 {result}")

    io.imshow(dilated)
    plt.title('Cleaned car')
    io.show()


def road_analysis():
    in_dir = "data/"
    im_name = "road.png"
    im_org = io.imread(in_dir + im_name)
    hsv_img = color.rgb2hsv(im_org)
    value_img = hsv_img[:, :, 2]

    segm_road = (value_img > 0.9)
    io.imshow(segm_road)
    plt.title('Segmented Road')
    io.show()

    label_img = measure.label(segm_road)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    areas = np.array([prop.area for prop in region_props])

    # Trick to sort descending
    areas_sort = -np.sort(-areas)
    print(f"Area 1 {areas_sort[0]} area 2 {areas_sort[1]}")
    answer = areas_sort[1]
    print(f"BLOBS with area less than {answer} should be removed")


def aorta_blob_analysis():
    in_dir = "data/"
    im_name = "1-442.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    t_1 = 90

    bin_img = (img > t_1)
    spleen_label_colour = color.label2rgb(bin_img)
    io.imshow(spleen_label_colour)
    plt.title("First aorta estimate")
    io.show()

    img_c_b = segmentation.clear_border(bin_img)
    label_img = measure.label(img_c_b)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)

    min_area = 200
    min_circ = 0.94

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        p = region.perimeter
        circ = 0
        if p > 0:
            circ = 4 * math.pi * a / (p * p)

        if p < 1 or a < min_area or circ < min_circ:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_aorta = label_img_filter > 0
    # show_comparison(img, i_area, 'Found spleen based on area')
    io.imshow(i_aorta)
    io.show()

    i_area = label_img_filter > 0
    pix_area = i_area.sum()
    one_pix = 0.75 * 0.75
    print(f"Number of pixels {pix_area} and {pix_area * one_pix:.0f} mm2")


def aorta_pixel_values():
    in_dir = "data/"
    im_name = "1-442.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    aorta_roi = io.imread(in_dir + 'AortaROI.png')
    aorta_mask = aorta_roi > 0
    aorta_values = img[aorta_mask]
    (mu_aorta, std_aorta) = norm.fit(aorta_values)
    print(f"Average {mu_aorta:.0f} standard deviation {std_aorta:.0f}")

    liver_roi = io.imread(in_dir + 'LiverROI.png')
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    (mu_liver, std_liver) = norm.fit(liver_values)

    min_hu = 147
    max_hu = 155
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)
    plt.plot(hu_range, pdf_aorta, 'r--', label="aorta")
    plt.plot(hu_range, pdf_liver, 'g', label="liver")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()
    # Answer = 151


def point_transformation():
    v = math.radians(20)
    a_scale = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    a_rot = np.array([[math.cos(v), -math.sin(v), 0], [math.sin(v), math.cos(v), 0],
                     [0, 0, 1]])
    a_trans = np.array([[1, 0, 3.1], [0, 1, -3.3], [0, 0, 1]])

    a_tot = np.matmul(a_trans, np.matmul(a_scale, a_rot))
    p = np.array([10, 10, 1])
    p_out = np.matmul(a_tot, p)
    print(p_out)


if __name__ == '__main__':
    # pca_on_football_data()
    # car_rgb_hsv_thresholds()
    # road_analysis()
    # aorta_analysis()
    # aorta_pixel_values()
    point_transformation()
