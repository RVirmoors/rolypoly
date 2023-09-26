import os
import zipfile

making = ''
osx_arm = False
osx_x86 = False
version = '0.2.11'

def zip_files_and_folders(input_list, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in input_list:
            if os.path.isfile(item):
                # If the item is a file, add it to the zip with the 'rolypoly' directory structure.
                arcname = os.path.join('rolypoly', os.path.basename(item))
                print(arcname)
                zipf.write(item, arcname=arcname)
            elif os.path.isdir(item):
                # If the item is a directory, add its contents to the zip with the 'rolypoly' directory structure.
                for root, _, files in os.walk(item):
                    newroot = root
                    if making == 'x86' and root.find('rolypoly~.mxo') != -1:
                        print("FOUND", root)
                        continue
                    elif making == 'arm' and root.find('rolypoly~_x86.mxo') != -1:
                        continue
                    if making == 'x86' and root.find('rolypoly~_x86.mxo'):
                        newroot = root.replace('rolypoly~_x86', 'rolypoly~')
                    for file in files:
                        arcname = os.path.join('rolypoly/py', os.path.relpath(os.path.join(newroot, file)))
                        # print("root: ", root)
                        print(arcname)
                        zipf.write(os.path.join(root, file), arcname=arcname)

if __name__ == "__main__":
    if os.path.isdir('../externals/rolypoly~_x86.mxo'):
        osx_x86 = True
    if os.path.isdir('../externals/rolypoly~.mxo'):
        osx_arm = True

    input_list = ['../docs',
            '../externals',
            '../extras',
            '../help',
            '../patchers',
            '../package-info.json',
            '../License.md',
            '../README.md',
            '../icon.png'
            ]
    output_zip = 'rolypoly-v' + version + '.zip' 

    if osx_arm or osx_x86:
        if osx_x86:
            making = 'x86'
            output_zip = 'rolypoly-v' + version + '_macOS_x64.zip'
            zip_files_and_folders(input_list, output_zip)
            print(f'Successfully created {output_zip}')
        if osx_arm:
            making = 'arm'
            output_zip = 'rolypoly-v' + version + '_macOS_arm64.zip'
            zip_files_and_folders(input_list, output_zip)
            print(f'Successfully created {output_zip}')
    else:
        making = 'win'
        input_list.append('../support')
        zip_files_and_folders(input_list, output_zip)
        print(f'Successfully created {output_zip}')

    
