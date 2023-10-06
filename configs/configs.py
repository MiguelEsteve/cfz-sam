import os

# Root folder for project code vs computer host
roots = {'DESKTOP-JV9DACD':
            {'cfz-sam': 'C:/repos/cfz-sam'},
         "PC-514445":
            {'cfz-sam': 'C:/repos/cfz-sam'}}
PROJECT_PATH = roots[os.getenv('computername')]['cfz-sam']
images_path = os.path.join(PROJECT_PATH, 'images')

