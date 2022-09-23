'''Create a list of 20x and 40x svs files based on the level 0 magnification'''
from pathlib import Path
import openslide

# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide')

# FUNCTIONS
def main():
    # Get a list of svs files
    svs_folders = [i for i in DATA_DIR.iterdir() for j in i.glob('*.svs') if j.is_file()]

    # Initialize Output Files
    print(f"folder", file=open('20x-list.txt','w'))
    print(f"folder", file=open('40x-list.txt','w'))
    print(f"folder,mag", file=open('misc-list.txt','w'))

    for i, folder in enumerate(svs_folders):
        # Read WSI
        wsi = next(folder.glob('*.svs'))

        # Read MetaData
        with openslide.OpenSlide(str(wsi)) as slide:
            # Add Name to appropriate output list
            if slide.properties.get('aperio.AppMag') == '20':
                # Write 20x
                print(f"{wsi.parent.name}", file=open('20x-list.txt','a'))
            elif slide.properties.get('aperio.AppMag') == '40':
                # Write 40x
                print(f"{wsi.parent.name}", file=open('40x-list.txt','a'))
            else:
                # Write left-overs
                print(f"{wsi.parent.name},{slide.properties.get('aperio.AppMag')}", file=open('misc-list.txt','a'))

        print(f"{i+1}/{len(svs_folders)}", end='\r')

# MAIN
if __name__ == "__main__":
    main()
