import os
import zipfile

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
                    for file in files:
                        arcname = os.path.join('rolypoly/py', os.path.relpath(os.path.join(root, file)))
                        print(arcname)
                        zipf.write(os.path.join(root, file), arcname=arcname)

if __name__ == "__main__":
    input_list = ['../docs',
                  '../externals',
                  '../extras',
                  '../help',
                  '../patchers',
                  '../support',
                  '../package-info.json',
                  '../License.md',
                  '../README.md',
                  '../icon.png'
                  ]
    output_zip = 'rolypoly-v0.2.11.zip' 

    zip_files_and_folders(input_list, output_zip)
    print(f'Successfully created {output_zip}')
