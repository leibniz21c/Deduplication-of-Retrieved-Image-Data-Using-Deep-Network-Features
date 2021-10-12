import sys

from dupimage.remover import remove_dup_images

def main(file_path, suffix=None, result_path=None):
    """
    pass
    """
    if suffix == None:
        suffix = 'jpg'
    
    remove_dup_images(
        root_dir=file_path,
        suffix=suffix,
        result_dir=result_path
    )

if __name__=="__main__":
    if len(sys.argv) == 4:
        main(
            file_path=sys.argv[1], 
            suffix=sys.argv[2], 
            result_path=sys.argv[3]
        )
    elif len(sys.argv) == 3:
        main(
            file_path=sys.argv[1], 
            suffix=sys.argv[2]
        )
    elif len(sys.argv) == 2:
        main(file_path=sys.argv[1])
    else:
        print("Usage: ") 
        print("%s [DIR_PATH]" % (sys.argv[0]))
        print("%s [DIR_PATH] [SUFFIX]" % (sys.argv[0]))
        print("%s [DIR_PATH] [SUFFIX] [RESULT_DIR_PATH]\n" % (sys.argv[0]))
