import os

# The name of your project
project_name = "my_project"

# The anonymous name to replace the project name with
anonymous_name = "anonymous_project"

# The directory of your local git repository
repo_directory = "/path/to/your/local/repo"

# A list of file extensions to search for
file_extensions = [".py", ".md", ".txt", ".html", ".css", ".js"]

# A function to replace project_name with anonymous_name in a file
def replace_project_name(file_path):
    with open(file_path, "r") as f:
        file_text = f.read()
    file_text = file_text.replace(project_name, anonymous_name)
    with open(file_path, "w") as f:
        f.write(file_text)

# Recursively search for files with the specified file extensions
for root, dirs, files in os.walk(repo_directory):
    for file in files:
        if file.endswith(tuple(file_extensions)):
            file_path = os.path.join(root, file)
            replace_project_name(file_path)