
import os
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import math

import os
import sys
import argparse

def add_stroke_pixels(center,width,stroke,gt,Is_fg):
    for i in range(-(width-2),width-1):
        for j in range(-(width-2),width-1):
            #check target pixel is within image range
            if((center[0]+i<gt.shape[0]) & (center[1]+j<gt.shape[1])):
                #check target pixel is a fg pixel or background pixel
                if((stroke.count([center[0]+i,center[1]+j])==0) & (gt[center[0]+i][center[1]+j]==Is_fg*255)):
                    stroke.append([center[0]+i,center[1]+j])
    return stroke

def random_move(image,seed_list,move_step,mean_stroke_length,mean_stroke_width,mask,Is_fg):
    strokes=[]
    MAX_RETRY = 100
    for seed in seed_list:
        retry_count=0
        print(seed)
        stroke_length = round(np.random.normal(mean_stroke_length, scale=10, size=None))
        stroke_width = round(np.random.normal(mean_stroke_width, scale=1, size=None))

        
        strokes = add_stroke_pixels(seed,stroke_width,strokes,mask,Is_fg)

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
            if((new_row<mask.shape[0]) & (new_col<mask.shape[1])):
                if(Is_fg):
                    
                    if((mask[new_row][new_col]>128)):
                        reject_cnt=0
                        current_row = new_row
                        current_col = new_col
                        length+=1
                        strokes = add_stroke_pixels([current_row,current_col],stroke_width,strokes,mask,Is_fg)
                        
                    else:
                        print('Debug: FG Step rejected:current',current_row,current_col,'new:',new_row,new_col,'mask:',mask[new_row][new_col],'iter:',it)
                        reject_cnt+=1
                        it=4
                        retry_count+=1
                        if(retry_count>MAX_RETRY):
                            f = open("ErrorList.txt", "a")
                            print("Debug: Skip and Log error in :{}".format(f))
                            f.write("Error:MAX_RETRY breached "+image+" \n")
                            f.close()
                            return -1
                        if(reject_cnt>=10):
                            if(len(strokes)==0):
                                ## report error and skip this image
                                f = open("ErrorList.txt", "a")
                                print("Debug: Skip and Log error in :{}".format(f))
                                f.write("Error:Random failed at the first seed "+image+" \n")
                                f.close()
                                
                            else:
                                current_location=strokes[np.random.randint(0,len(strokes))]
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
                        strokes = add_stroke_pixels([current_row,current_col],stroke_width,strokes,mask,Is_fg)
                    else:
                        print('Debug: BG Step rejected:current',current_row,current_col,'new:',new_row,new_col,'mask:',mask[new_row][new_col],'iter:',it)
                        reject_cnt+=1
                        it=4
                        retry_count+=1
                        if(retry_count>MAX_RETRY):
                            f = open("ErrorList.txt", "a")
                            print("Debug: Skip and Log error in :{}".format(f))
                            f.write("Error:MAX_RETRY breached "+image+" \n")
                            f.close()
                            break
                        if(reject_cnt>=10):
                            if(len(strokes)==0):
                                ## report error and skip this image
                                f = open("ErrorList.txt", "a")
                                print("Debug: Skip and Log error in :{}".format(f))
                                f.write("Error:Random failed at the first seed "+image+" \n")
                                f.close()
                                return -1
                            else:
                                current_location=strokes[np.random.randint(0,len(strokes))]
                                print('Debug: changing current location from:,', current_row,current_col,'to:',current_location)
                                current_row=current_location[0]
                                current_col=current_location[1]
                                reject_cnt=0
    return strokes

def edge_(mask,num_edge_seeds_fg,num_edge_seeds_bg):
    edges = cv2.Canny(mask,100,200)
    kernel = np.ones((10,10),np.uint8)
    edge = cv2.dilate(edges,kernel,iterations = 1)
    edge_pixels_fg=[]
    edge_pixels_bg=[]
    stroke_edge_fg = []
    stroke_edge_bg = []
    
    for x in range (edge.shape[0]):
        for y in range (edge.shape[1]):
            if (edge[x][y]==255) & (mask[x][y]>128):
                edge_pixels_fg.append([x,y])
            if (edge[x][y]==255) & (mask[x][y]<128):
                edge_pixels_bg.append([x,y])
    rand_edge=np.random.randint(0,len(edge_pixels_fg),num_edge_seeds_fg)
    edge_seed_fg = [edge_pixels_fg[sd] for sd in rand_edge]
    
    rand_edge=np.random.randint(0,len(edge_pixels_bg),num_edge_seeds_bg)
    edge_seed_bg = [edge_pixels_bg[sd] for sd in rand_edge]
    
    #generate edge fg and bg pixels from edge seeds
    for seed in edge_seed_fg:
        x1 = seed[0] - 30 if(seed[0] - 30>=0) else 0
        x2 = seed[0] + 30 if(seed[0] + 30<mask.shape[0]) else mask.shape[0]
        y1 = seed[1] - 30 if(seed[1] - 30>=0) else 0
        y2 = seed[1] + 30 if(seed[1] + 30<mask.shape[1]) else mask.shape[1]
        window = edge[x1:x2,y1:y2]
        mask_window = mask[x1:x2,y1:y2]
        for index,i in np.ndenumerate(window):
            if ((window[index]==255) & (mask_window[index]>200)):
                stroke_edge_fg.append([seed[0]-30+index[0],seed[1]-30+index[1]])

    for seed in edge_seed_bg:
        x1 = seed[0] - 30 if(seed[0] - 30>=0) else 0
        x2 = seed[0] + 30 if(seed[0] + 30<mask.shape[0]) else mask.shape[0]
        y1 = seed[1] - 30 if(seed[1] - 30>=0) else 0
        y2 = seed[1] + 30 if(seed[1] + 30<mask.shape[1]) else mask.shape[1]
        window = edge[x1:x2,y1:y2]
        mask_window = mask[x1:x2,y1:y2]
        for index,i in np.ndenumerate(window):
            if((window[index]==255) & (mask_window[index]<100)):
                stroke_edge_bg.append([seed[0]-30+index[0],seed[1]-30+index[1]])
    return stroke_edge_fg,stroke_edge_bg

def arg_parse():
    parser = argparse.ArgumentParser(description="arguments parser")
    parser.add_argument("--i", dest="Image_Dir", default='./DUTS-TR/DUTS-TR-Image/',help="Image directory for Datasets.")
    parser.add_argument("--m", dest="Mask_Dir", default='./DUTS-TR/DUTS-TR-Mask/', help="Mask directory for Datasets.")
    parser.add_argument("--o", dest="Out_Dir", default='./DUTS-TR/Ouput/',help="Output directory")
    
    parser.add_argument("--positive_brush", dest="positive_brush", default=[0,255,0],help="positive_brush")
    parser.add_argument("--negative_brush", dest="negative_brush", default=[255,0,0],help="negative_brush")
    return parser.parse_args()


def main():

    prog_args = arg_parse()
    
    Image_Dir = prog_args.Image_Dir
    Mask_Dir = prog_args.Mask_Dir
    Out_Dir = prog_args.Out_Dir
    positive_brush = prog_args.positive_brush
    negative_brush = prog_args.negative_brush

    # Read in images
    images = [f for f in glob.glob(Image_Dir+'*.jpg', recursive=True)]
    masks = [f for f in glob.glob(Mask_Dir+'*.png', recursive=True)]
    current_interation_maps = [f for f in glob.glob(Out_Dir+'InterationMaps/fg/'+'*.png', recursive=True)]

    for f in images:
        print('processing:',f)
        
        ## initialize random variables
        n_samples = np.random.randint(5,10)   
        mean_stroke_length = np.random.randint(30,90)
        mean_stroke_width = np.random.randint(3,9)
        move_step = 1
        num_bg_seeds_n_samples=np.random.randint(1,6,n_samples)
        num_fg_seeds_n_samples=np.random.randint(1,6,n_samples)
        num_edge_seeds_fg_n_samples=np.random.randint(0,6,n_samples)
        num_edge_seeds_bg_n_samples=np.random.randint(0,6,n_samples)
        
        rand_parms = zip(num_fg_seeds_n_samples,num_bg_seeds_n_samples,num_edge_seeds_fg_n_samples,num_edge_seeds_bg_n_samples)
        
        for iter,(num_fg_seeds,num_bg_seeds,num_edge_seeds_fg,num_edge_seeds_bg) in enumerate(rand_parms):
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
            stroke_edge_fg,stroke_edge_bg = edge_(mask,num_edge_seeds_fg,num_edge_seeds_bg)


            ## generate foreground and background strokes pixels
            foreground_pixels=[]
            background_pixels=[]
            for x in range (mask.shape[0]):
                for y in range (mask.shape[1]):
                    if mask[x][y]==255:
                        foreground_pixels.append([x,y])
                    else:
                        background_pixels.append([x,y])

            rand_fg=np.random.randint(0,len(foreground_pixels),num_fg_seeds)
            rand_bg=np.random.randint(0,len(background_pixels),num_bg_seeds)

            seeds_fg = [foreground_pixels[sd] for sd in rand_fg]
            seeds_bg = [background_pixels[sd] for sd in rand_bg]

            stroke_fg = random_move(f,seeds_fg,move_step,mean_stroke_length,mean_stroke_width,mask,1)
            stroke_bg = random_move(f,seeds_bg,move_step,mean_stroke_length,mean_stroke_width,mask,0)

            if((stroke_fg==-1) or (stroke_bg==-1)):
                # Move image to error folder
                filename = mask_filename.split('/')[-1]
                
                if(not os.path.exists(Out_Dir+"Error/")):
                    os.makedirs(Out_Dir+"Error/") 
                cv2.imwrite( Out_Dir+"Error/"+filename, image )
                continue
            
            #combine fg_edges with fg and bg_edges with bg
            stroke_fg = stroke_fg + stroke_edge_fg
            stroke_bg = stroke_bg + stroke_edge_bg


            fg_mask=np.zeros(mask.shape,dtype=np.uint8)
            bg_mask=np.zeros(mask.shape,dtype=np.uint8)            
            ## overlay original image for visualization
            if(len(stroke_fg)>0):
                fg_stroke = np.array(stroke_fg)
                fg_row_num = fg_stroke[0:,:1]
                fg_col_num = fg_stroke[0:,1:]
                image[fg_row_num,fg_col_num] = positive_brush
                fg_mask[fg_row_num,fg_col_num]=1
            if(len(stroke_bg)>0):
                bg_stroke = np.array(stroke_bg)
                bg_row_num = bg_stroke[0:,:1]
                bg_col_num = bg_stroke[0:,1:]
                image[bg_row_num,bg_col_num] = negative_brush
                bg_mask[bg_row_num,bg_col_num]=1

            


            # generate L2 distance map
            fg_mask_invert = 1-fg_mask
            dismap_fg = cv2.distanceTransform(fg_mask_invert, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            bg_mask_invert = 1-bg_mask
            dismap_bg = cv2.distanceTransform(bg_mask_invert, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

            if(not os.path.exists(Out_Dir+"/Images_with_strokes/")):
                Dir = Out_Dir+"Images_with_strokes/"
                os.makedirs(Dir) 
                print("Directory '% s' created" % Dir)
            if(not os.path.exists(Out_Dir+'InterationMaps/fg/')):
                Dir = Out_Dir+'InterationMaps/fg/'
                os.makedirs(Dir) 
                print("Directory '% s' created" % Dir) 
            if(not os.path.exists(Out_Dir+'InterationMaps/bg/')):
                Dir = Out_Dir+'InterationMaps/bg/'
                os.makedirs(Dir) 
                print("Directory '% s' created" % Dir)
                
            filename = mask_filename.split('/')[-1]
            name = filename.split('.')[0]
            extension = filename.split('.')[1]
            name = name+'_{}'.format(iter)
            filename = name+"."+extension
            
            cv2.imwrite( Out_Dir+"Images_with_strokes/"+filename, image )
            cv2.imwrite( Out_Dir+'InterationMaps/fg/'+filename, dismap_fg )
            cv2.imwrite( Out_Dir+'InterationMaps/bg/'+filename, dismap_bg )
            print("saved image_with_strokes to: "+Out_Dir+"/Images_with_strokes/"+filename)
            print("saved foreground interaction map to: "+Out_Dir+'InterationMaps/fg/'+filename)
            print("saved background interaction map to: "+Out_Dir+'InterationMaps/bg/'+filename)



if __name__ == '__main__':
    main()