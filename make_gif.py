import imageio
import glob
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


with imageio.get_writer('gan.gif', mode='I', duration=.1) as writer:
    filenames = glob.glob('outputs/overfit/overfit_*')
    filenames = sorted(filenames, key=natural_keys)
    print(filenames)
    print(len(filenames))
    last = -1
    for i,filename in enumerate(filenames):
        if i % 2 == 0:
            image = imageio.imread(filename)
            writer.append_data(image)
        #print(i)
        if i == len(filenames)-1:
            print('adding extra')
            for _ in range(15):
                writer.append_data(image)
