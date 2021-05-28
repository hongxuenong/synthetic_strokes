## Updates 2020.05.19
New method for stroke creation.\
Sample images:\
![Image description](https://github.com/hongxuenong/synthetic_strokes/blob/master/Test/Output/Images_with_strokes/0a0c62c3f6c4606adae355de1fd551d4-hflip_0.png)
![Image description](https://github.com/hongxuenong/synthetic_strokes/blob/master/Test/Output/Images_with_strokes/0a0c62c3f6c4606adae355de1fd551d4-hflip_3.png)

## map_utils
**Example usages**:\
import map_utils\
    fg_map,bg_map = map_utils.extract_strokes(image) ##default: fg=2,bg=1\
    fg_map,bg_map = map_utils.extract_strokes(image,fg=2,bg=1) ##implicitly specify fg and bg value
    
### Usage
**Single Core Version**

run "python generate.py"

Specify dataset directory by "-i", default location is './DUTS-TR'

Specify output directory by "-o",default location is './DUTS-TR/Output'

Each image is processed 5-10 times with different:

*  number of foreground strokes;

*  number of background strokes;

*  length of strokes;


**Multiprocessing**

run "python generate_multi.py"

Each image is ensured to have at least one foreground stroke and background stroke.

### Known issue:
**


