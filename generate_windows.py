import os
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import math

def add_fg_stroke_pixels(center,width,stroke,gt):
    for i in range(-(width-2),width-1):
        for j in range(-(width-2),width-1):
            if((center[0]+i<gt.shape[0]) & (center[1]+j<gt.shape[0])):
                if((stroke.count([center[0]+i,center[1]+j])==0) & (gt[center[0]+i][center[1]+j]==255)):
                    stroke.append([center[0]+i,center[1]+j])
    return stroke
def add_bg_stroke_pixels(center,width,stroke,gt):
    for i in range(-(width-2),width-1):
        for j in range(-(width-2),width-1):
            if((center[0]+i<gt.shape[0]) & (center[1]+j<gt.shape[1])):
                if((stroke.count([center[0]+i,center[1]+j])==0) & (gt[center[0]+i][center[1]+j]!=255)):
                    stroke.append([center[0]+i,center[1]+j])
    return stroke

def random_move(seed_list,mean_stroke_length,mean_stroke_width,mask,Is_fg):
    strokes=[]
    for seed in seed_list:
        print(seed)
        stroke_length = round(np.random.normal(mean_stroke_length, scale=10, size=None))
        stroke_width = round(np.random.normal(mean_stroke_width, scale=1, size=None))

        strokes = add_bg_stroke_pixels(seed,stroke_width,strokes,mask)
        current_row = seed[0]
        current_col = seed[1]
        length=0
        direction = [0,0]
        while(direction[0]==0&direction[1]==0):
            direction = np.random.randint(-1,2,2)
        it=0
        reject_cnt=0
        while(length<stroke_length):
            it+=1
            if(it>=5):
                
                new_direction = np.random.randint(-1,2,2)
                while((new_direction[0]==0 & new_direction[1]==0)or(new_direction[0]==direction[0]&new_direction[1]==direction[1])):
                    new_direction = np.random.randint(-1,2,2)
                direction = new_direction
                it=0
            new_row = current_row+move_step*direction[0]
            new_col = current_col+move_step*direction[1]
            if((new_row<image.shape[0]) & (new_col<image.shape[1])):
                if(Is_fg):
                    
                    if((mask[new_row][new_col]>128)):
                        reject_cnt=0
                        current_row = new_row
                        current_col = new_col
                        length+=1
                        strokes = add_fg_stroke_pixels([current_row,current_col],stroke_width,strokes,mask)
                        
                    else:
                        print('Debug: FG Step rejected:current',current_row,current_col,'new:',new_row,new_col,'mask:',mask[new_row][new_col],'iter:',it)
                        reject_cnt+=1
                        it=4
                        if(reject_cnt>=10):
#                             print(len(strokes))
                            index = np.random.randint(0,len(strokes))
                            current_location=strokes[index]
                            print('Debug: changing current location from:,', current_row,current_col,'to:',current_location)
                            current_row=current_location[0]
                            current_col=current_location[1]
                            reject_cnt=0
                else:
                    if((mask[new_row][new_col]<128)):
                        reject_cnt=0
                        current_row = new_row
                        current_col = new_col
                        length+=1
                        strokes = add_bg_stroke_pixels([current_row,current_col],stroke_width,strokes,mask)
                    else:
                        print('Debug: BG Step rejected:current',current_row,current_col,'new:',new_row,new_col,'mask:',mask[new_row][new_col],'iter:',it)
                        reject_cnt+=1
                        it=4
                        if(reject_cnt>=10):
#                             print(len(strokes))
                            current_location=strokes[np.random.randint(0,len(strokes))]
                            print('Debug: changing current location from:,', current_row,current_col,'to:',current_location)
                            current_row=current_location[0]
                            current_col=current_location[1]
                            reject_cnt=0
    return strokes



mean_stroke_length = 60
mean_stroke_width = 6
move_step = 1
num_bg_seeds=np.random.randint(1,6)
num_fg_seeds=np.random.randint(1,6)
num_edge_seeds_fg=np.random.randint(0,4)
num_edge_seeds_bg=np.random.randint(0,4)
num_edge_seeds=np.random.randint(1,3)
positive_brush = [0,255,0]
negative_brush = [255,0,0]


Image_Dir = './DUTS-TR/DUTS-TR-Image/'
Mask_Dir = './DUTS-TR/DUTS-TR-Mask/'
Map_Dir = './DUTS-TR/InteractionMaps/fg/'
# file1 = open("Output.txt","a") 

images = [f for f in glob.glob(Image_Dir+'*.jpg', recursive=True)]
masks = [f for f in glob.glob(Mask_Dir+'*.png', recursive=True)]
current_interation_maps = [f for f in glob.glob(Map_Dir+'*.png', recursive=True)]

for f in images:
    print('processing:',f)
    Map_filename = f.replace('jpg','png')
    Map_filename = Map_filename.replace('DUTS-TR-Image','InteractionMaps/fg')
    if any(Map_filename in s for s in current_interation_maps):
        print(Map_filename,' is already processed. Skipping...')
        continue
    mask_filename = f.replace('jpg','png')
    mask_filename = mask_filename.replace('Image','Mask')
    print(mask_filename)

    image = cv2.imread(f)
    mask = cv2.imread(mask_filename,0)

    ## detect edges of objects
    edges = cv2.Canny(mask,100,200)
    kernel = np.ones((10,10),np.uint8)
    edge = cv2.dilate(edges,kernel,iterations = 1)
    edge_pixels_fg=[]
    for x in range (edge.shape[0]):
        for y in range (edge.shape[1]):
            if edge[x][y]==255:
                edge_pixels_fg.append([x,y])
    rand_edge=np.random.randint(0,len(edge_pixels_fg),num_edge_seeds_fg)
    edge_seed_fg = [edge_pixels_fg[sd] for sd in rand_edge]
#     print(edge_seed_fg)

    edge_pixels_bg=[]
    for x in range (edge.shape[0]):
        for y in range (edge.shape[1]):
            if edge[x][y]==255:
                edge_pixels_bg.append([x,y])
    rand_edge=np.random.randint(0,len(edge_pixels_bg),num_edge_seeds_bg)
    edge_seed_bg = [edge_pixels_bg[sd] for sd in rand_edge]
#     print(edge_seed_bg)

    ##get foreground and background seeds
    foreground_pixels=[]
    background_pixels=[]

    for x in range (mask.shape[0]):
        for y in range (mask.shape[1]):
            if mask[x][y]==255:
                foreground_pixels.append([x,y])
            else:
                background_pixels.append([x,y])
    # print(foreground_pixels)
    # print(background_pixels)
    rand_fg=np.random.randint(0,len(foreground_pixels),num_fg_seeds)
#     print(rand_fg)

    rand_bg=np.random.randint(0,len(background_pixels),num_bg_seeds)
#     print(rand_bg)


    seeds_fg = [foreground_pixels[sd] for sd in rand_fg]
    seeds_bg = [background_pixels[sd] for sd in rand_bg]
#     print(seeds_fg)
#     print(seeds_bg)


    stroke_fg = random_move(seeds_fg,mean_stroke_length,mean_stroke_width,mask,1)

    stroke_bg = random_move(seeds_bg,mean_stroke_length,mean_stroke_width,mask,0)
    
    stroke_edge_fg = []
    # mask_new=mask;
    stroke_edge_bg = []

    for seed in edge_seed_fg:
        x1 = seed[0] - 30 if(seed[0] - 30>=0) else 0
        x2 = seed[0] + 30 if(seed[0] + 30<edge.shape[0]) else edge.shape[0]
        y1 = seed[1] - 30 if(seed[1] - 30>=0) else 0
        y2 = seed[1] + 30 if(seed[1] + 30<edge.shape[1]) else edge.shape[1]
        window = edge[x1:x2,y1:y2]
        mask_window = mask[x1:x2,y1:y2]
        for index,i in np.ndenumerate(window):
            if ((window[index]==255) & (mask_window[index]>200)):
                stroke_edge_fg.append([seed[0]-30+index[0],seed[1]-30+index[1]])

    for seed in edge_seed_bg:
        x1 = seed[0] - 30 if(seed[0] - 30>=0) else 0
        x2 = seed[0] + 30 if(seed[0] + 30<edge.shape[0]) else edge.shape[0]
        y1 = seed[1] - 30 if(seed[1] - 30>=0) else 0
        y2 = seed[1] + 30 if(seed[1] + 30<edge.shape[1]) else edge.shape[1]
        window = edge[x1:x2,y1:y2]
        mask_window = mask[x1:x2,y1:y2]
        for index,i in np.ndenumerate(window):
            if((window[index]==255) & (mask_window[index]<100)):
                stroke_edge_bg.append([seed[0]-30+index[0],seed[1]-30+index[1]])

                
    ## overlay original image
    fg_stroke = np.array(stroke_fg)
    fg_row_num = fg_stroke[0:,:1]
    fg_col_num = fg_stroke[0:,1:]
    image[fg_row_num,fg_col_num] = [0,255,0]
    # print(image[row_num, col_num])

    bg_stroke = np.array(stroke_bg)
    bg_row_num = bg_stroke[0:,:1]
    bg_col_num = bg_stroke[0:,1:]
    image[bg_row_num,bg_col_num] = [255,0,0]

    if(num_edge_seeds_fg!=0):
        stroke_edge_fg = np.array(stroke_edge_fg)
        edgefg_row_num = stroke_edge_fg[0:,:1]
        edgefg_col_num = stroke_edge_fg[0:,1:]
        image[edgefg_row_num,edgefg_col_num] = positive_brush
    if(num_edge_seeds_bg!=0):
        stroke_edge_bg = np.array(stroke_edge_bg)
        edgebg_row_num = stroke_edge_bg[0:,:1]
        edgebg_col_num = stroke_edge_bg[0:,1:]
        image[edgebg_row_num,edgebg_col_num] = negative_brush

    filename = mask_filename.split('\\')[-1]
    
    fg_map = np.ones([mask.shape[0],mask.shape[1]])*255
    bg_map = np.ones([mask.shape[0],mask.shape[1]])*255
    
    for sp in stroke_fg:
    #     print('processing:',sp)
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):

                d = math.sqrt((i-sp[0])**2+(j-sp[1])**2)
                if(d>255):
                    d=255
                if(fg_map[i][j]>d):
                    fg_map[i][j]=d


    for sp in stroke_bg:
    #     print('processing:',sp)
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):

                d = math.sqrt((i-sp[0])**2+(j-sp[1])**2)
                if(d>255):
                    d=255
                if(bg_map[i][j]>d):
                    bg_map[i][j]=d
    for sp in stroke_edge_fg:
    #     print('processing:',sp)
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):

                d = math.sqrt((i-sp[0])**2+(j-sp[1])**2)
                if(d>255):
                    d=255
                if(fg_map[i][j]>d):
                    fg_map[i][j]=d


    for sp in stroke_edge_bg:
    #     print('processing:',sp)
        for i in range(mask.shape[0]):
            for j in range (mask.shape[1]):

                d = math.sqrt((i-sp[0])**2+(j-sp[1])**2)
                if(d>255):
                    d=255
                if(bg_map[i][j]>d):
                    bg_map[i][j]=d

    cv2.imwrite( "./DUTS-TR/Images_with_strokes/"+filename, image )
    cv2.imwrite( "./DUTS-TR/InteractionMaps/fg/"+filename, fg_map )
    cv2.imwrite( "./DUTS-TR/InteractionMaps/bg/"+filename, fg_map )
    print("saved image_with_strokes to: "+"./DUTS-TR/Images_with_strokes/"+filename)
    # file1.write("saved image_with_strokes to: "+"./DUTS-TR/Images_with_strokes/"+filename+'\n')
    print("saved foreground interaction map to: "+"./DUTS-TR/InteractionMaps/fg/"+filename)
    print("saved background interaction map to: "+"./DUTS-TR/InteractionMaps/bg/"+filename)
