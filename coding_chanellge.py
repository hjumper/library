import numpy as np
import cv2

PX2UMFACOTR = 1.4 # factor for the transformation from pixel to um
MINAREA = 10 # minimum area for the cell

def display_img(im, name:str="default", resize_im:bool=True) -> None:
    '''Display image'''
    
    if resize_im: im = cv2.resize(im, (889, 738))
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def detect_and_count_cells(im:np.ndarray, kernel_size:int=5, open_iter_nums:int=2, erode_iter_nums:int=4)-> tuple:
    '''detect and count cells'''
    
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    # adaptive binarization + inversion + stratch removal
    _, b_im =  cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # fill holes
    im_cells = cv2.morphologyEx(b_im, cv2.MORPH_OPEN, kernel, iterations=open_iter_nums)
    # over erosion for cell detection
    im_cells_erotion = cv2.erode(im_cells,kernel,iterations = erode_iter_nums)
    # calculate number of cells
    retval, _, _, _ = cv2.connectedComponentsWithStats(im_cells_erotion, connectivity=8)
    
    num_cells = retval - 1 # background

    return im_cells, num_cells

def draw_contours(b_im:np.ndarray, linewidth:int=4)->tuple: # linewidth should equal to erode_iter_nums argument in function detect_and_count_cells
    '''draw contours of cells'''
    
    contours, _ = cv2.findContours(b_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = tuple(filter(lambda cnt:cv2.contourArea(cnt)>MINAREA, contours))
    cnt_im = np.zeros_like(b_im, dtype=np.uint8)
    cv2.drawContours(cnt_im, contours, -1, 255, linewidth)

    display_img(cnt_im)
    return contours

def retrieve_avg_dim(contours:tuple, num_cells:int)-> tuple:
    '''return the average height and width of cells'''
    
    width,height = 0,0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        width += rect[1][0]
        height += rect[1][1]

    return PX2UMFACOTR*height/num_cells, PX2UMFACOTR*width/num_cells


# 3556 2953 image size
if __name__ == '__main__':
    im = cv2.imread('printing_cylinder_surface.tif', cv2.IMREAD_GRAYSCALE)
    im_cells, num_cells = detect_and_count_cells(im)
    display_img(im_cells)
    contours=draw_contours(im_cells)    
    avg_h, avg_w = retrieve_avg_dim(contours, num_cells=num_cells)
