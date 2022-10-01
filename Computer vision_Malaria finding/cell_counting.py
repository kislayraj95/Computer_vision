import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel

# This function is provided to you. You will need to call it.
# You should not need to modify it.
def seedfill(im, seed_row, seed_col, fill_color, bckg):
    """
    im: The image on which to perform the seedfill algorithm
    seed_row and seed_col: position of the seed pixel
    fill_color: Color for the fill
    bckg: Color of the background, to be filled
    Returns: Number of pixels filled
    Behavior: Modifies image by performing seedfill
    """
    size=0  # keep track of patch size
    n_row, n_col = im.shape
    front={(seed_row,seed_col)}  # initial front
    while len(front)>0:
        r, c = front.pop()  # remove an element from front
        if im[r, c]==bckg: 
            im[r, c]=fill_color  # color the pixel
            size+=1
            # look at all neighbors
            for i in range(max(0,r-1), min(n_row,r+2)):
                for j in range(max(0,c-1),min(n_col,c+2)):
                    # if background, add to front
                    if im[i,j]==bckg and\
                       (i,j) not in front:
                        front.add((i,j))
    return size

# QUESTION 1
def detect_edges(image_color):
    """
    Args:
        image_color: the original color image read from an image_file
    Returns:         
        image_graytone: a grayscale image converted from the original image
        image_sobel: a new image of detected edges
    """
    # WRITE YOUR CODE HERE
    image_graytone = rgb2gray(image_color) # converting to grayscale
    image_sobel = sobel(image_graytone) # applying the Sobel filter
    return image_graytone, image_sobel

# QUESTION 2
def binarize_edges(image_sobel1):
    """
    Args:
        image_sobel1: an ndarray as the initial edge detected image from Q1
    Returns: 
        image_sobel2: a new image where the pixels whose value grear or equal to 
        0.05 are set to 1; othewise to 0
    """    
    # WRITE YOUR CODE HERE
    image_sobel2 = np.where(image_sobel1>0.05, 1.0, 0.0) # binarizing the image
    return image_sobel2

# QUESTION 3
def cleanup(image_graytone, image_sobel2):
    """
    Args:
        image_graytone: the grayscale image from Q1
        image_sobel2: the image from Q2 where pixel value is either 0 or 1
    Returns: 
        image_sobel3: a modified version of image_sobel2, where any white pixel at 
        position (r,c) is replaced by a black pixel if the value of pixel (r,c) 
        or of any of the 8 surrounding pixels in image_graytone image is below 0.5. 
    """    
    # WRITE YOUR CODE HERE
    nrow, ncol = image_sobel2.shape # getting the dimensions of the image
    for r in range(nrow): # iterating over the rows
        for c in range(ncol): # iterating over the columns
            if (image_graytone[r, c])<0.5: # checking if the pixel is below 0.5
                image_sobel2[r, c] = 0.0 # setting the pixel to 0
            else: # if the pixel is above 0.5
                if r-1 in range(nrow): # checking if the r is in the range
                    if image_graytone[r-1, c]<0.5: # checking if the pixel is below 0.5
                        image_sobel2[r, c] = 0.0 # setting the pixel to 0
                    if c-1 in range(nrow): # checking if the column is in the range
                        if image_graytone[r, c-1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                        elif image_graytone[r-1, c-1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                    if c+1 in range(nrow): # checking if the column is in the range
                        if image_graytone[r, c+1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                        elif image_graytone[r-1, c+1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0 
                if r+1 in range(nrow): # checking if the r is in the range
                    if image_graytone[r+1, c]<0.5: # checking if the pixel is below 0.5
                        image_sobel2[r, c] = 0.0 # setting the pixel to 0
                    if c+1 in range(ncol): # checking if the column is in the range
                        if image_graytone[r, c+1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                        elif image_graytone[r+1, c+1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                    if c-1 in range(ncol): # checking if the column is in the range
                        if image_graytone[r, c-1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
                        elif image_graytone[r+1, c-1]<0.5: # checking if the pixel is below 0.5
                            image_sobel2[r, c] = 0.0 # setting the pixel to 0
    return image_sobel2

# QUESTION 4
def fill_cells(image_sobel3):
    """
    Args:
        edge_image: A black-and-white image, with black background and
                    white edges
    Returns: 
        filled_image: A new image where each close region is filled with 
                    a different grayscale value
    """
    # WRITE YOUR CODE HERE
    filled_image = image_sobel3.copy() # copying the image because it somehow gave an error if I didn't
    nrow, ncol = filled_image.shape # getting the dimensions of the image
    fill_colour = 0.5 
    seedfill(filled_image, 0, 0, 0.1, 0.0) 
    for r in range(nrow):
        for c in range(ncol):
            if filled_image[r, c] == 0.0:
                seedfill(filled_image, r, c, fill_colour, 0.0) 
                fill_colour += 0.001
    return filled_image

# QUESTION 5
def classify_cells(image_graytone, filled_image, \
                   min_size=1000, max_size=5000, \
                   infected_grayscale=0.5, min_infected_percentage=0.02):
    """
    Args:
        image_graytone: The graytone image from Q1
        filled_image: A graytone image, with each closed region colored
                       with a different grayscal value from Q4
        min_size, max_size: 
            The min and max size of a region to be called a cell
        infected_grayscale: 
            Maximum grayscale value for a pixel to be called infected
        min_infected_percentage: 
            Smallest fraction of dark pixels needed to call a cell infected
    Returns: A tuple of two sets, containing the grayscale values of cells 
             that are infected and not infected
    """
    # WRITE YOUR CODE HERE
    nrow, ncol = image_graytone.shape # getting the dimensions of the image
    all_greyscale_values = set() # creating a set to store all the grayscale values
    for r in range(nrow):
        for c in range(ncol):
            # Adding the grayscale values from filled_image to the set
            all_greyscale_values.add(filled_image[r, c])
    infected = set() # creating a set to store the infected grayscale values
    not_infected = set() # creating a set to store the not infected grayscale values
    for value in all_greyscale_values: # iterating over the set
        g_count = 0 # to count the number of pixels with the same grayscale value
        infected_count = 0 # to count the number of pixels with infected grayscale value
        for r in range(nrow):
            for c in range(ncol):     
                if filled_image[r, c] == value: # checking if the grayscale value is the same
                    g_count += 1 # incrementing the g_count
                    if image_graytone[r, c] <= infected_grayscale: # checking if the pixel is infected
                        infected_count += 1 # incrementing the infected count
        # checking if the area with the grayscale value can be considered a cell using size
        if g_count >= min_size and g_count <= max_size: 
            if (g_count*min_infected_percentage) > infected_count: # checking if the percentage of infected pixels is greater than the minimum percentage
                not_infected.add(value)
            elif (g_count*min_infected_percentage) <= infected_count: # checking if the percentage of infected pixels is less than or equal to the minimum percentage
                infected.add(value)
    return infected, not_infected

    # QUESTION 6
def annotate_image(color_image, filled_image, infected, not_infected):
    """
    Args:
        image_color: the original color image read from image_file in Q1
        filled_image: A graytone image, with each closed region colored
                       with a different grayscal value in Q4
        infected: A set of graytone values of infected cells
        not_infected: A set of graytone values of non-infcted cells
    Returns: A color image, with infected cells highlighted in red
             and non-infected cells highlighted in green
    """    
    # WRITE YOUR CODE HERE
    new_colour_image = color_image.copy() # copying the image because it somehow gave an error if I didn't
    nrow, ncol = filled_image.shape # getting the dimensions of the image
    for r in range(nrow):
        for c in range(ncol):
            if filled_image[r, c] in infected: # checking if the grayscale value is infected
                if r-1 in range(nrow): # checking if the row is in the range
                    if filled_image[r-1, c] == 1.0:
                        new_colour_image[r, c] = [255, 0, 0]
                    if c-1 in range(nrow): # checking if the column is in the range
                        if filled_image[r, c-1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                        elif filled_image[r-1, c-1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                    if c+1 in range(nrow): # checking if the column is in the range
                        if filled_image[r, c+1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                        elif filled_image[r-1, c+1]<0.5: 
                            new_colour_image[r, c] = [255, 0, 0]      
                if r+1 in range(nrow): # checking if the row is in the range
                    if filled_image[r+1, c] == 1.0:
                        new_colour_image[r, c] = [255, 0, 0]
                    if c+1 in range(ncol): # checking if the column is in the range
                        if filled_image[r, c+1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                        elif filled_image[r+1, c+1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                    if c-1 in range(ncol): # checking if the column is in the range
                        if filled_image[r, c-1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]
                        elif filled_image[r+1, c-1] == 1.0:
                            new_colour_image[r, c] = [255, 0, 0]  
            elif filled_image[r, c] in not_infected: # checking if the grayscale value is not infected
                if r-1 in range(nrow): # checking if the row is in the range
                    if filled_image[r-1, c] == 1.0:
                        new_colour_image[r, c] = [0, 255, 0]
                    if c-1 in range(nrow): # checking if the column is in the range
                        if filled_image[r, c-1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                        elif filled_image[r-1, c-1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                    if c+1 in range(nrow): # checking if the column is in the range
                        if filled_image[r, c+1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                        elif filled_image[r-1, c+1]<0.5:
                            new_colour_image[r, c] = [0, 255, 0]
                if r+1 in range(nrow): # checking if the row is in the range
                    if filled_image[r+1, c] == 1.0:
                        new_colour_image[r, c] = [0, 255, 0]
                    if c+1 in range(ncol): # checking if the column is in the range
                        if filled_image[r, c+1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                        elif filled_image[r+1, c+1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                    if c-1 in range(ncol): # checking if the column is in the range
                        if filled_image[r, c-1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
                        elif filled_image[r+1, c-1] == 1.0:
                            new_colour_image[r, c] = [0, 255, 0]
    return new_colour_image

# MAIN FUNCTION
if __name__ == "__main__":  # do not remove this line
    
    # QUESTION 1 TEST CODE
    # The path of the malaria image file. Default to "malaria-1.jpg"
    # Change it to "malaria-1-small.jpg" for quick debugging    
    image_file = "malaria-1.jpg"
    # image_file = "malaria-1-small.jpg"
    
    image_color = io.imread(image_file)
    image_graytone,image_sobel1 = detect_edges(image_color)    
    io.imsave("Q1_gray_sobel.jpg",image_sobel1)
    
    # QUESTION 2 TEST CODE
    image_sobel2 = binarize_edges(image_sobel1)
    io.imsave("Q2_gray_sobel_T005.jpg",image_sobel2)
    # print(image_sobel2.shape)
    # print(image_color.shape)
    # # QUESTION 3 TEST CODE
    image_sobel3 = cleanup(image_graytone, image_sobel2)
    io.imsave("Q3_gray_sobel_T005_cleanup.jpg",image_sobel3)
    
    # # QUESTION 4 TEST CODE
    image_filled = fill_cells(image_sobel3)
    io.imsave("Q4_gray_sobel_T005_cleanup_filled.jpg",image_filled)
    
    # # QUESTION 5 TEST CODE
    infected, not_infected = classify_cells(image_graytone, image_filled)
    print(infected)
    print(not_infected)
    
    # # QUESTION 6 TEST CODE
    annotated_image = annotate_image(image_color, image_filled, 
                                     infected, not_infected)
    io.imsave("Q6_annotated.jpg", annotated_image)
    